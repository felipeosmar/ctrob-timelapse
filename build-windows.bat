@echo off
echo ========================================
echo  CTROB Timelapse - Build Windows
echo ========================================
echo.

pip install PySide6 pyinstaller
if errorlevel 1 (
    echo ERRO: Falha ao instalar dependencias. Verifique se o Python esta no PATH.
    pause
    exit /b 1
)

echo.
echo Gerando executavel...
python build.py --onefile
if errorlevel 1 (
    echo ERRO: Falha no build.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Build concluido!
echo  Executavel: dist\ctrob-timelapse.exe
echo ========================================
pause
