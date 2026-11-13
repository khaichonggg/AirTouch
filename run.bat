@echo off
rem Runs the gesture app with its camera window. Extra words are passed on: run.bat --lite
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :run
echo 还没有安装好。请先双击 setup.bat，装好之后再打开这个文件。
echo.
pause
exit /b 1

:run
".venv\Scripts\python.exe" hand_mouse.py %*
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" exit /b 0
echo.
echo 程序出了问题，已经停止。上面的文字就是原因。
pause
exit /b %CODE%
