@echo off
rem Runs the gesture self-check and keeps the window open to read the result.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :run
echo 还没有安装好。请先双击 setup.bat，装好之后再打开这个文件。
echo.
pause
exit /b 1

:run
".venv\Scripts\python.exe" check_gestures.py %*
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%
