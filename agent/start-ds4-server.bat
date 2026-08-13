@echo off
setlocal
for %%I in ("%~dp0..") do set "REPO_DIR=%%~fI"

echo Checking ds4-server build...
rem Make performs an incremental build: it recompiles only when the binary or
rem one of its source dependencies is missing or newer.  The local RTX 3090
rem uses compute capability 8.6.
wsl.exe --cd "%REPO_DIR%" make --no-print-directory -j2 ds4-server CUDA_ARCH=sm_86
if errorlevel 1 (
    echo ERROR: ds4-server build failed; the server was not started.
    pause
    exit /b 1
)

echo Starting ds4-server on http://127.0.0.1:8080 ...
echo CUDA decode baseline: Q8_1-R8, split-K32 and GQA2
rem WSL NAT forwards this guest listener to Windows localhost.
wsl.exe --cd "%REPO_DIR%" ./ds4-server --cuda -m gguf/Qwen3.6-27B-Q4_K_S.gguf --ctx 32768 --host 0.0.0.0 --port 8080 %*

pause
