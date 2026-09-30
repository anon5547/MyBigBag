@echo off
setlocal
cd /d "%~dp0"
set "PYEXE="
for %%V in (3.12 3.13 3.11 3.10) do if not defined PYEXE py -%%V -c "import sys" >nul 2>nul && set "PYEXE=py -%%V"
if not defined PYEXE set "PYEXE=python"
%PYEXE% "%~dp0setup.py" --uninstall
