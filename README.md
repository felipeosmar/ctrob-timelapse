# CTROB Timelapse

Organizador de fotos timelapse de cartão SD com interface gráfica. Copia e renomeia fotos automaticamente, e gera vídeo timelapse com ffmpeg.

## Funcionalidades

- **Copiar e organizar fotos** — lê a árvore de pastas do cartão SD e copia renomeando por data e sequência
- **Gerar vídeo timelapse** — cria MP4 a partir das fotos com resolução e FPS configuráveis
- **Barra de progresso** — acompanhe em tempo real a cópia e geração do vídeo
- **Multiplataforma** — Windows e Linux
- **Portável** — sem instalação, basta copiar e executar

## Estrutura do Cartão SD

O software espera a seguinte estrutura de pastas (câmera timelapse):

```
SD/
├── 2025-08-12/
│   └── 001/
│       └── jpg/
│           └── 20/
│               └── 00.08[R][0@0][0].jpg
├── 2025-08-22/
│   └── 001/
│       └── jpg/
│           ├── 06/
│           │   └── 00.00[R][0@0][0].jpg
│           ├── 07/
│           │   └── 00.00[R][0@0][0].jpg
│           └── .../
```

## Resultado

As fotos são copiadas para a pasta destino com nomes padronizados:

```
Destino/
├── 2025-08-12_001.jpg
├── 2025-08-22_001.jpg
├── 2025-08-22_002.jpg
├── ...
└── timelapse.mp4
```

## Requisitos

- **Python 3.10+**
- **ffmpeg** (para geração do vídeo)

## Uso Rápido (a partir do código fonte)

```bash
# Instalar dependências
pip install PySide6

# Executar
python main.py
```

## Build Portável

### Linux

```bash
python -m venv venv
source venv/bin/activate
pip install PySide6 pyinstaller
python build.py
```

Resultado: `dist/ctrob-timelapse/` — copie a pasta inteira.

### Windows

Duplo clique em `build-windows.bat` ou:

```
pip install PySide6 pyinstaller
python build.py --onefile
```

Resultado: `dist\ctrob-timelapse.exe` — arquivo único portável.

## Configurações do Timelapse

| Opção | Padrão | Valores |
|-------|--------|---------|
| Resolução | 1920x1080 | 3840x2160, 1280x720, 1080x1080 |
| FPS | 25 | 1 a 120 |

## Instalação do ffmpeg

- **Linux:** `sudo apt install ffmpeg`
- **Windows:** `winget install ffmpeg` ou baixe em [ffmpeg.org](https://ffmpeg.org/download.html)

## Autor

Criado por **Felipe O. de Aviz** (felipe.aviz@sc.senai.br)
