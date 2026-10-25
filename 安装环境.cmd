@echo off
cd /d "%~dp0"
py -3.12 setup_environment.py --with-llm
pause
