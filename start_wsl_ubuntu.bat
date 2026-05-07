@echo off
REM WSL Ubuntu + Docker Desktop başlat
echo =============================================
echo  WSL Ubuntu + Docker Desktop Baslat
echo =============================================
echo.

REM Docker Desktop baslat
echo [1/3] Docker Desktop baslatiliyor...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo  Baslatti, bekleniyor...
timeout /t 10 /nobreak >nul

REM WSL Ubuntu baslat
echo [2/3] WSL Ubuntu baslatiliyor...
wsl -d Ubuntu
echo.

echo =============================================
echo  Hazir! WSL'de su komutlari kullanabilirsin:
echo.
echo  cd /mnt/c/Users/BAHADIR/Desktop/nanovllm-voxcpm-official
echo  python voxcpm2_run.py
echo.
echo  Docker icin:
echo  docker build -f Dockerfile.inference -t voxcpm2:latest .
echo  docker run --gpus all voxcpm2:latest
echo =============================================
pause
