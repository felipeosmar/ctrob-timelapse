"""
CTROB Timelapse — Organizador de fotos timelapse de cartão SD.

Copia fotos de uma árvore de pastas (cartão SD de câmera timelapse)
para uma pasta destino, renomeando com data e sequência.
Gera vídeo timelapse a partir das fotos copiadas.

Estrutura esperada do SD:
    {data}/{001}/jpg/{hora}/{arquivo}.jpg

Resultado:
    {destino}/{data}_{seq:03d}.jpg
    {destino}/timelapse.mp4
"""

import shutil
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class CopyWorker(QObject):
    """Worker que executa a cópia em thread separada."""

    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(int, int)  # copied, errors
    error = Signal(str)

    def __init__(self, source: Path, destination: Path):
        super().__init__()
        self.source = source
        self.destination = destination
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            photos = self._scan_photos()
            if not photos:
                self.error.emit("Nenhuma foto encontrada na origem.")
                return

            self.destination.mkdir(parents=True, exist_ok=True)

            copied = 0
            errors = 0
            total = len(photos)

            for i, (src_path, new_name) in enumerate(photos):
                if self._cancel:
                    break

                dst_path = self.destination / new_name
                try:
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                except Exception as e:
                    errors += 1
                    self.progress.emit(i + 1, total, f"ERRO: {new_name} — {e}")
                    continue

                self.progress.emit(i + 1, total, new_name)

            self.finished.emit(copied, errors)

        except Exception as e:
            self.error.emit(str(e))

    def _scan_photos(self) -> list[tuple[Path, str]]:
        """
        Varre a árvore do SD e retorna lista de (caminho_original, novo_nome).

        Estrutura: {data}/{???}/jpg/{hora}/{arquivo}.jpg
        Resultado: {data}_{seq:03d}.jpg
        """
        photos: list[tuple[Path, str]] = []

        date_dirs = sorted(
            d
            for d in self.source.iterdir()
            if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"
        )

        for date_dir in date_dirs:
            date_str = date_dir.name
            seq = 1

            jpg_files = sorted(date_dir.rglob("*.jpg"))

            for jpg_path in jpg_files:
                new_name = f"{date_str}_{seq:03d}.jpg"
                photos.append((jpg_path, new_name))
                seq += 1

        return photos


class TimelapseWorker(QObject):
    """Worker que gera o vídeo timelapse via ffmpeg."""

    progress = Signal(int, int, str)  # current, total, msg
    finished = Signal(str)  # output path
    error = Signal(str)

    def __init__(
        self,
        photos_dir: Path,
        output_path: Path,
        fps: int = 25,
        resolution: str = "1920x1080",
    ):
        super().__init__()
        self.photos_dir = photos_dir
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            # Listar fotos em ordem
            photos = sorted(self.photos_dir.glob("*.jpg"))
            if not photos:
                self.error.emit("Nenhuma foto encontrada na pasta destino.")
                return

            total = len(photos)
            self.progress.emit(0, total, f"Preparando {total} fotos...")

            # Criar arquivo de lista para ffmpeg concat
            list_file = self.photos_dir / "_ffmpeg_list.txt"
            with open(list_file, "w") as f:
                for photo in photos:
                    f.write(f"file '{photo.name}'\n")
                    f.write(f"duration {1/self.fps:.6f}\n")
                # Repetir última para duração correta
                f.write(f"file '{photos[-1].name}'\n")

            self.progress.emit(0, total, "Gerando vídeo com ffmpeg...")

            # Executar ffmpeg
            w, h = self.resolution.split("x")
            vf = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:-1:-1:color=black,"
                f"setsar=1"
            )
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-r",
                str(self.fps),
                str(self.output_path),
            ]

            cmd.extend(["-progress", "pipe:1", "-stats_period", "0.5"])

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.photos_dir),
            )

            # Monitorar progresso via stdout (-progress pipe:1)
            stderr_data = []
            current_frame = 0

            def _read_stderr():
                for line in iter(process.stderr.readline, b""):
                    stderr_data.append(
                        line.decode("utf-8", errors="replace").strip()
                    )

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()

            for line in iter(process.stdout.readline, b""):
                if self._cancel:
                    process.kill()
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str.startswith("frame="):
                    try:
                        current_frame = int(line_str.split("=")[1])
                        pct = min(current_frame, total)
                        self.progress.emit(
                            pct, total, f"Frame {current_frame}/{total}"
                        )
                    except (IndexError, ValueError):
                        pass

            process.wait()
            stderr_thread.join(timeout=2)

            # Limpar arquivo temporário
            list_file.unlink(missing_ok=True)

            if self._cancel:
                self.error.emit("Cancelado pelo usuário.")
                return

            if process.returncode != 0:
                err = "\n".join(stderr_data[-10:])
                self.error.emit(f"ffmpeg erro (code {process.returncode}):\n{err}")
                return

            duration_s = total / self.fps
            minutes = int(duration_s // 60)
            seconds = int(duration_s % 60)
            size_mb = self.output_path.stat().st_size / (1024 * 1024)

            self.progress.emit(total, total, "Concluído!")
            self.finished.emit(
                f"✅ Vídeo gerado: {self.output_path.name}\n"
                f"   {total} frames, {minutes}m{seconds:02d}s, {size_mb:.1f} MB"
            )

        except FileNotFoundError:
            self.error.emit(
                "ffmpeg não encontrado. Instale com: sudo apt install ffmpeg"
            )
        except Exception as e:
            self.error.emit(str(e))


BUTTON_GREEN = (
    "QPushButton { background-color: #4CAF50; color: white; font-size: 14px; "
    "font-weight: bold; border-radius: 6px; }"
    "QPushButton:hover { background-color: #45a049; }"
    "QPushButton:disabled { background-color: #ccc; color: #666; }"
)

BUTTON_BLUE = (
    "QPushButton { background-color: #2196F3; color: white; font-size: 14px; "
    "font-weight: bold; border-radius: 6px; }"
    "QPushButton:hover { background-color: #1976D2; }"
    "QPushButton:disabled { background-color: #ccc; color: #666; }"
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CTROB Timelapse")
        self.setMinimumSize(650, 550)
        self.worker = None
        self.thread = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # === Origem ===
        layout.addWidget(QLabel("<b>📂 Origem (Cartão SD):</b>"))
        row_src = QHBoxLayout()
        self.src_input = QLineEdit()
        self.src_input.setPlaceholderText("Selecione o cartão SD...")
        self.src_input.setReadOnly(True)
        row_src.addWidget(self.src_input)
        btn_src = QPushButton("Selecionar")
        btn_src.clicked.connect(self._select_source)
        row_src.addWidget(btn_src)
        layout.addLayout(row_src)

        # === Destino ===
        layout.addWidget(QLabel("<b>📁 Destino:</b>"))
        row_dst = QHBoxLayout()
        self.dst_input = QLineEdit()
        self.dst_input.setPlaceholderText("Selecione a pasta de destino...")
        self.dst_input.setReadOnly(True)
        row_dst.addWidget(self.dst_input)
        btn_dst = QPushButton("Selecionar")
        btn_dst.clicked.connect(self._select_destination)
        row_dst.addWidget(btn_dst)
        layout.addLayout(row_dst)

        # === Botão Copiar ===
        row_copy = QHBoxLayout()
        self.btn_copy = QPushButton("▶  Copiar Fotos")
        self.btn_copy.setMinimumHeight(40)
        self.btn_copy.setStyleSheet(BUTTON_GREEN)
        self.btn_copy.clicked.connect(self._start_copy)
        row_copy.addWidget(self.btn_copy)

        self.btn_cancel_copy = QPushButton("⏹  Cancelar")
        self.btn_cancel_copy.setMinimumHeight(40)
        self.btn_cancel_copy.setEnabled(False)
        self.btn_cancel_copy.clicked.connect(self._cancel_task)
        row_copy.addWidget(self.btn_cancel_copy)
        layout.addLayout(row_copy)

        # === Timelapse ===
        grp_tl = QGroupBox("🎬 Gerar Timelapse")
        tl_layout = QVBoxLayout(grp_tl)

        row_tl_cfg = QHBoxLayout()
        row_tl_cfg.addWidget(QLabel("Resolução:"))
        self.cmb_resolution = QComboBox()
        self.cmb_resolution.addItems(
            ["1920x1080", "3840x2160", "1280x720", "1080x1080"]
        )
        row_tl_cfg.addWidget(self.cmb_resolution)

        row_tl_cfg.addWidget(QLabel("FPS:"))
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(1, 120)
        self.spn_fps.setValue(25)
        row_tl_cfg.addWidget(self.spn_fps)

        row_tl_cfg.addStretch()
        tl_layout.addLayout(row_tl_cfg)

        row_tl_btn = QHBoxLayout()
        self.btn_timelapse = QPushButton("🎬  Gerar Vídeo")
        self.btn_timelapse.setMinimumHeight(40)
        self.btn_timelapse.setStyleSheet(BUTTON_BLUE)
        self.btn_timelapse.clicked.connect(self._start_timelapse)
        row_tl_btn.addWidget(self.btn_timelapse)

        self.btn_cancel_tl = QPushButton("⏹  Cancelar")
        self.btn_cancel_tl.setMinimumHeight(40)
        self.btn_cancel_tl.setEnabled(False)
        self.btn_cancel_tl.clicked.connect(self._cancel_task)
        row_tl_btn.addWidget(self.btn_cancel_tl)
        tl_layout.addLayout(row_tl_btn)

        layout.addWidget(grp_tl)

        # === Progresso ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Aguardando...")
        layout.addWidget(self.lbl_status)

        # === Log ===
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        layout.addWidget(self.log)

        # === Rodapé ===
        lbl_footer = QLabel("Criado por Felipe O. de Aviz (felipe.aviz@sc.senai.br)")
        lbl_footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_footer.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        lbl_footer.setCursor(Qt.CursorShape.IBeamCursor)
        lbl_footer.setStyleSheet("color: #999; font-size: 11px; margin-top: 4px;")
        layout.addWidget(lbl_footer)

    def _select_source(self):
        path = QFileDialog.getExistingDirectory(self, "Selecionar Cartão SD")
        if path:
            self.src_input.setText(path)

    def _select_destination(self):
        path = QFileDialog.getExistingDirectory(self, "Selecionar Pasta de Destino")
        if path:
            self.dst_input.setText(path)

    def _set_buttons_busy(self, busy: bool):
        self.btn_copy.setEnabled(not busy)
        self.btn_timelapse.setEnabled(not busy)
        self.btn_cancel_copy.setEnabled(busy)
        self.btn_cancel_tl.setEnabled(busy)

    def _start_copy(self):
        src = self.src_input.text().strip()
        dst = self.dst_input.text().strip()

        if not src or not dst:
            QMessageBox.warning(self, "Atenção", "Selecione origem e destino.")
            return

        source = Path(src)
        destination = Path(dst)

        if not source.exists():
            QMessageBox.warning(self, "Erro", "Pasta de origem não encontrada.")
            return

        self._set_buttons_busy(True)
        self.log.clear()
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Escaneando fotos...")

        self.worker = CopyWorker(source, destination)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_copy_finished)
        self.worker.error.connect(self._on_error)

        self.thread = threading.Thread(target=self.worker.run, daemon=True)
        self.thread.start()

    def _start_timelapse(self):
        dst = self.dst_input.text().strip()
        if not dst:
            QMessageBox.warning(self, "Atenção", "Selecione a pasta destino com fotos.")
            return

        photos_dir = Path(dst)
        if not photos_dir.exists() or not list(photos_dir.glob("*.jpg")):
            QMessageBox.warning(
                self, "Atenção", "Nenhuma foto .jpg na pasta destino. Copie primeiro."
            )
            return

        output_path = photos_dir / "timelapse.mp4"
        resolution = self.cmb_resolution.currentText()
        fps = self.spn_fps.value()

        self._set_buttons_busy(True)
        self.log.clear()
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Preparando timelapse...")

        self.worker = TimelapseWorker(photos_dir, output_path, fps, resolution)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_timelapse_finished)
        self.worker.error.connect(self._on_error)

        self.thread = threading.Thread(target=self.worker.run, daemon=True)
        self.thread.start()

    def _cancel_task(self):
        if self.worker:
            self.worker.cancel()
            self.lbl_status.setText("Cancelando...")

    def _on_progress(self, current: int, total: int, msg: str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f"{current}/{total} — {msg}")
        self.log.append(msg)

    def _on_copy_finished(self, copied: int, errors: int):
        self._set_buttons_busy(False)
        msg = f"✅ Cópia concluída! {copied} fotos copiadas"
        if errors:
            msg += f", {errors} erros"
        self.lbl_status.setText(msg)
        self.log.append(f"\n{msg}")
        QMessageBox.information(self, "Cópia Concluída", msg)

    def _on_timelapse_finished(self, result: str):
        self._set_buttons_busy(False)
        self.lbl_status.setText("Timelapse gerado!")
        self.log.append(f"\n{result}")
        QMessageBox.information(self, "Timelapse Concluído", result)

    def _on_error(self, error_msg: str):
        self._set_buttons_busy(False)
        self.lbl_status.setText(f"❌ Erro: {error_msg}")
        QMessageBox.critical(self, "Erro", error_msg)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
