@echo off
rem Puts the on/off switch in the tray, with no console window.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" goto :run
echo 还没有安装好。请先双击 setup.bat，装好之后再打开这个文件。
echo.
pause
exit /b 1

:run
start "" ".venv\Scripts\pythonw.exe" "airtouch_tray.pyw"
