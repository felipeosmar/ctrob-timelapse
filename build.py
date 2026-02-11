"""
Build script — gera executável portável com PyInstaller.

Uso:
    python build.py          # build para o SO atual
    python build.py --clean  # limpa builds anteriores antes

Resultado:
    dist/ctrob-timelapse/    # pasta portável, copiar inteira
"""

import os
import platform
import shutil
import subprocess
import sys


def main():
    clean = "--clean" in sys.argv

    if clean:
        for d in ["build", "dist"]:
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"Removido: {d}/")

    # Verificar PyInstaller
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Instalando PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Detectar SO
    system = platform.system()
    print(f"Build para: {system} ({platform.machine()})")

    # Nome do executável
    exe_name = "ctrob-timelapse"
    if system == "Windows":
        exe_name = "ctrob-timelapse.exe"

    mode = "--onefile" if "--onefile" in sys.argv else "--onedir"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=ctrob-timelapse",
        mode,
        "--windowed",
        "--noconfirm",
        "--clean",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtGui",
        "--hidden-import=PyQt5.QtWidgets",
        "main.py",
    ]

    print(f"Executando: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    print()
    print("=" * 50)
    print(f"✅ Build concluído!")
    print(f"   Pasta portável: dist/ctrob-timelapse/")
    print(f"   Executável:     dist/ctrob-timelapse/{exe_name}")
    print()
    print("Copie a pasta 'dist/ctrob-timelapse/' inteira para qualquer máquina.")
    if system != "Windows":
        print("⚠️  Para gerar .exe Windows, rode este script em uma máquina Windows.")
    print("⚠️  ffmpeg precisa estar instalado na máquina de destino.")


if __name__ == "__main__":
    main()
