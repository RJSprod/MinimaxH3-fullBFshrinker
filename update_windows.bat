@echo off
rem ===========================================================================
rem  MiniMax H3 Checkpoint Converter - updater
rem
rem  Pulls the latest project revision (if this is a git checkout), re-syncs the
rem  locked dependency set, and re-runs the self-tests. Your logs/ folder and
rem  converted checkpoints are never touched.
rem ===========================================================================

setlocal enabledelayedexpansion
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
cd /d "%PROJECT_ROOT%"

set "UV_VERSION=0.12.3"
set "UV_DIR=%PROJECT_ROOT%\installer_files\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%PROJECT_ROOT%\installer_files\python"
set "UV_CACHE_DIR=%PROJECT_ROOT%\installer_files\uv-cache"
set "UV_TOOL_DIR=%PROJECT_ROOT%\installer_files\uv-tools"
set "VIRTUAL_ENV=%PROJECT_ROOT%\.venv"

if not exist "%UV_EXE%" (
    echo uv is not installed yet. Run start_windows.bat once first.
    pause
    endlocal
    exit /b 1
)

if exist "%PROJECT_ROOT%\.git" (
    echo [1/4] Updating project source...
    git pull --ff-only
    if errorlevel 1 (
        echo       git pull failed - resolve the conflict manually, then re-run.
        pause
        endlocal
        exit /b 1
    )
) else (
    echo [1/4] Not a git checkout - skipping source update.
    echo       Download the latest release ZIP to update the source.
)

echo [2/4] Re-syncing locked dependencies...
"%UV_EXE%" sync --frozen 2>nul
if errorlevel 1 (
    echo       Lock changed - re-resolving...
    "%UV_EXE%" sync
    if errorlevel 1 goto :fail
)

echo [3/4] Running environment self-tests...
"%UV_EXE%" run python -m h3converter.bootstrap_checks
if errorlevel 1 goto :fail

echo [4/4] Installed versions:
"%UV_EXE%" run python -m h3converter.bootstrap_checks --versions-only

echo.
echo Update complete.
pause
endlocal
exit /b 0

:fail
echo.
echo Update failed. See the messages above and the logs\ folder.
pause
endlocal
exit /b 1
