@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

<<<<<<< HEAD
if not exist ".\.venv39\Scripts\activate.bat" (
    echo Error: .venv39 not found in this folder. Check venv path.
    pause
    exit /b 1
)
call ".\.venv39\Scripts\activate.bat"

cd "%~dp0stereo-calibration"
=======
if not exist "%~dp0..\.venv39\Scripts\activate.bat" (
    echo Error: .venv39 not found in parent folder. Check venv path.
    pause
    exit /b 1
)
call "%~dp0..\.venv39\Scripts\activate.bat"
>>>>>>> master
echo Running: python -m calib_v3.main %*

python -m calib_v3.main %*
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% neq 0 (
    echo.
    echo Exit code: %EXIT_CODE%
)
echo.
pause
endlocal
exit /b %EXIT_CODE%

