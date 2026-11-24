@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YOLO 标注训练工具

if not exist ".venv\Scripts\python.exe" (
    echo [提示] 未检测到运行环境，正在启动安装程序...
    echo 若安装未自动开始，请先双击运行 setup.bat
    call setup.bat
    if errorlevel 1 exit /b 1
)

if not exist "logs" mkdir logs

".venv\Scripts\python.exe" -m app.main 2>> logs\stderr.log
if errorlevel 1 (
    echo.
    echo [错误] 程序异常退出，请查看上方错误信息或 logs 目录下的日志文件
    pause
)
