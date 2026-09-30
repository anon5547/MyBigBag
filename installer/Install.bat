@echo off
setlocal
cd /d "%~dp0"
title OmniBrain Setup
REM ASCII only on purpose: cmd.exe mangles UTF-8. All Thai text lives in setup.py (Tk window).
REM Find Python 3.10+ . The "py" launcher goes first because plain "python" may be the Microsoft Store stub.
set "PYEXE="
for %%V in (3.12 3.13 3.11 3.10) do if not defined PYEXE py -%%V -c "import sys" >nul 2>nul && set "PYEXE=py -%%V"
if not defined PYEXE python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" >nul 2>nul && set "PYEXE=python"
if defined PYEXE goto run

echo Python 3.10 or newer was not found. Installing Python 3.12 with winget ...
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
REM This window still has the old PATH, but the py launcher and the default folder work right away.
if not defined PYEXE py -3.12 -c "import sys" >nul 2>nul && set "PYEXE=py -3.12"
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if defined PYEXE goto run

echo.
echo Could not install Python automatically.
echo Install Python 3.12 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
echo then double-click Install.bat again.
pause
exit /b 1

:run
echo Starting the OmniBrain installer ...
%PYEXE% "%~dp0setup.py" %*
if errorlevel 1 (
  echo.
  echo Setup did not finish. See the messages above.
  pause
)
exit /b %errorlevel%
