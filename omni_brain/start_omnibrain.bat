@echo off
REM Double-click to start OmniBrain on Windows 10/11. First run installs the dependencies.
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python app.py
pause
