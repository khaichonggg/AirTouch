@echo off
rem One-click install: finds Python, then lets install.py do the rest.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ===== AirTouch 一键安装 =====
echo.

rem 3.12 first: the version AirTouch is built and tested on. Asking Python to run a line
rem of code makes the Microsoft Store's fake "python" fail instead of opening the Store,
rem so it can never be picked by mistake.
set "PY="
call :try_python py -3.12
if defined PY goto :found
call :try_python py -3
if defined PY goto :found
call :try_python python
if defined PY goto :found

echo 这台电脑上还没有 Python。请按下面的步骤装一下，只需要做一次：
echo.
echo   1. 打开这个网址：https://www.python.org/downloads/windows/
echo   2. 找到 Python 3.12，下载 Windows installer 64-bit
echo   3. 运行下载好的文件。第一个画面最下面的 Add python.exe to PATH 一定要打勾
echo   4. 点 Install Now，等它装完
echo   5. 关掉这个窗口，再双击一次 setup.bat
echo.
pause
exit /b 1

:found
call %PY% install.py
set "CODE=%ERRORLEVEL%"
echo.
if "%CODE%"=="0" goto :done
echo 这次没有装好。看看上面的提示，处理之后再双击一次 setup.bat 就可以了。
echo.
:done
pause
exit /b %CODE%

:try_python
rem "call" so that a python.bat shim (as pyenv installs) returns here instead of ending this script.
call %* -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>&1
if "%ERRORLEVEL%"=="0" set "PY=%*"
exit /b 0
