@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (echo Run the environment installer first. & pause & exit /b 1)
".venv\Scripts\python.exe" run.py
pause
