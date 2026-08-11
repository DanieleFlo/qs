@echo off
echo Starting ds4-server Inference Engine via WSL on http://127.0.0.1:8000 ...

set "ABS_ROOT_DIR=%~dp0"

echo Target Directory: %ABS_ROOT_DIR%

wsl --cd "%ABS_ROOT_DIR%" chmod +x ./ds4-server 2>nul
wsl --cd "%ABS_ROOT_DIR%" ./ds4-server --ctx 100000 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192

pause
