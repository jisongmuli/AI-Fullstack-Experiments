@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\server-control.ps1" -Action Start
if errorlevel 1 goto failed
start "" "http://127.0.0.1:8002/"
:failed
echo.
pause
