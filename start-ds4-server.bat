@echo off
echo Starting ds4-server Inference Engine via WSL on http://127.0.0.1:8080 ...

set "ABS_ROOT_DIR=%~dp0"

echo Target Directory: %ABS_ROOT_DIR%

echo Checking ds4-server build...
wsl --cd "%ABS_ROOT_DIR%" make --no-print-directory -j2 ds4-server CUDA_ARCH=sm_86
if errorlevel 1 (
    echo ERROR: ds4-server build failed; the server was not started.
    pause
    exit /b 1
)

echo CUDA decode baseline: Qwen3.8 target-only, Q8_1-R8, split-K32 and GQA2
wsl --cd "%ABS_ROOT_DIR%" ./ds4-server --cuda -m gguf/Qwen3.8-27B-UD-Q4_K_S.gguf --ctx 22593 --host 0.0.0.0 --port 8080 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 8192 %*

pause
