@echo off
rem ===========================================================================
rem  MiniMax H3 Checkpoint Converter - Windows launcher
rem
rem  Double-click this file. It bootstraps everything:
rem     pinned uv  ->  pinned Python 3.12  ->  project .venv  ->  self-tests  ->  GUI
rem
rem  You do NOT need a preinstalled Python, and you never activate the venv
rem  yourself. Dependency decisions live in pyproject.toml / uv.lock, not here.
rem ===========================================================================

setlocal enabledelayedexpansion
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
cd /d "%PROJECT_ROOT%"

rem Pinned uv standalone release. Bump deliberately, never automatically.
set "UV_VERSION=0.12.3"
set "UV_DIR=%PROJECT_ROOT%\installer_files\uv"
set "UV_EXE=%UV_DIR%\uv.exe"

rem Keep the managed Python and uv caches inside the project so the install
rem stays self-contained and uninstalling is just deleting the folder.
set "UV_PYTHON_INSTALL_DIR=%PROJECT_ROOT%\installer_files\python"
set "UV_CACHE_DIR=%PROJECT_ROOT%\installer_files\uv-cache"
set "UV_TOOL_DIR=%PROJECT_ROOT%\installer_files\uv-tools"
set "VIRTUAL_ENV=%PROJECT_ROOT%\.venv"

if not exist "%UV_EXE%" call :install_uv || goto :fail

echo [1/3] Syncing environment (this can take several minutes on first run)...
"%UV_EXE%" sync --frozen 2>nul
if errorlevel 1 (
    rem No lockfile yet, or the lock is out of date relative to pyproject.toml.
    echo       Lockfile missing or stale - resolving dependencies...
    "%UV_EXE%" sync || goto :fail
)

echo [2/3] Running environment self-tests...
"%UV_EXE%" run python -m h3converter.bootstrap_checks
if errorlevel 1 goto :checks_failed

echo [3/3] Starting MiniMax H3 Converter...
"%UV_EXE%" run python -m h3converter.app %*
if errorlevel 1 goto :fail

endlocal
exit /b 0


:install_uv
echo [0/3] Downloading pinned uv %UV_VERSION%...
if not exist "%UV_DIR%" mkdir "%UV_DIR%"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;" ^
  "Invoke-WebRequest -Uri '%UV_URL%' -OutFile '%UV_DIR%\uv.zip';" ^
  "Expand-Archive -Path '%UV_DIR%\uv.zip' -DestinationPath '%UV_DIR%' -Force;" ^
  "Remove-Item '%UV_DIR%\uv.zip' -Force"
if errorlevel 1 (
    echo.
    echo   Could not download uv from:
    echo     %UV_URL%
    echo   Check your internet connection or a corporate proxy/firewall, then retry.
    exit /b 1
)
if not exist "%UV_EXE%" (
    echo   uv.exe was not found after extraction - the download may be corrupt.
    echo   Delete "%UV_DIR%" and run this file again.
    exit /b 1
)
exit /b 0


:checks_failed
echo.
echo ===========================================================================
echo  Environment self-tests FAILED. Conversion was not started.
echo  The full diagnostic is in:  %PROJECT_ROOT%\logs\
echo  Scroll up for the specific check that failed.
echo ===========================================================================
echo.
pause
endlocal
exit /b 1


:fail
echo.
echo ===========================================================================
echo  Startup failed. See the messages above and the logs\ folder.
echo ===========================================================================
echo.
pause
endlocal
exit /b 1
