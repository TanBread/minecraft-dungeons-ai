@echo off
title Minecraft Dungeons AI - Installer
echo ========================================
echo   Minecraft Dungeons AI - Installer
echo ========================================
echo.

:: Check for admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Install VC++ Redistributable
echo [1/3] Installing Visual C++ Redistributable...
where vcredist >nul 2>&1
set "VCREDIST_DONE=0"

:: Try winget first
winget install Microsoft.VCRedist.2015+.x64 --accept-source-agreements --accept-package-agreements >nul 2>&1
if %errorlevel% equ 0 (
    set "VCREDIST_DONE=1"
    echo   Done.
)

:: Fallback: download and install manually
if "%VCREDIST_DONE%"=="0" (
    echo   Trying direct download...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\vc_redist.x64.exe'}" >nul 2>&1
    if exist "%TEMP%\vc_redist.x64.exe" (
        "%TEMP%\vc_redist.x64.exe" /install /quiet /norestart >nul 2>&1
        del "%TEMP%\vc_redist.x64.exe" >nul 2>&1
        echo   Done.
    ) else (
        echo   WARNING: Could not install VC++ Redistributable.
        echo   Download manually from: https://aka.ms/vs/17/release/vc_redist.x64.exe
    )
)

:: Check for CUDA
echo [2/3] Checking CUDA runtime...
if exist "%SYSTEMROOT%\System32\cublas64_13.dll" (
    echo   CUDA 13 found.
) else if exist "%PROGRAMFILES%\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\cublas64_13.dll" (
    echo   CUDA 13 found.
) else (
    echo   WARNING: CUDA 13.3 not found.
    echo   The agent requires CUDA 13.3 runtime (cublas64_13.dll, curand64_10.dll).
    echo   Install from: https://developer.nvidia.com/cuda-downloads
    echo.
)

:: Check for NVIDIA driver
echo [3/3] Checking NVIDIA driver...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>nul
) else (
    echo   WARNING: nvidia-smi not found. Ensure NVIDIA GPU drivers are installed.
)

echo.
echo ========================================
echo   Installation complete!
echo.
echo   To run training:  run.bat
echo   To run headless:  run_noviewer.bat (logs to train_log.txt)
echo ========================================
echo.
pause
