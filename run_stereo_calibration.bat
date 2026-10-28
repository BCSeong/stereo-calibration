@echo off
setlocal enabledelayedexpansion
rem ----------------------------------------------------------------------------
rem  Stereo calibration launcher (Windows)
rem  Usage : double-click, or  run_stereo_calibration.bat [arguments]
rem  Python lookup order:
rem    1) .venv  inside this folder          (recommended, see README "Setup")
rem    2) ..\.venv39  next to this folder    (shared lota_tools environment)
rem    3) py -3 / python on PATH
rem ----------------------------------------------------------------------------
cd /d "%~dp0"

set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set PY="%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\.venv39\Scripts\python.exe" set PY="%~dp0..\.venv39\Scripts\python.exe"
if not defined PY (
    where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Error: Python not found.
    echo Create a virtual environment in this folder first:
    echo     py -3.9 -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Python : !PY!
echo Running: -m calib_v3.main %*
echo.
!PY! -m calib_v3.main %*
set EXIT_CODE=!ERRORLEVEL!
if !EXIT_CODE! neq 0 (
    echo.
    echo Exit code: !EXIT_CODE!
)
echo.
pause
endlocal & exit /b %EXIT_CODE%
