@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title YOLO 标注训练工具 - 环境安装

echo ==========================================================
echo   YOLO 标注训练工具 - 一键环境安装
echo   仅需运行一次，安装完成后双击 start.bat 即可使用
echo ==========================================================
echo.

REM ---------- 1. 查找可用的 Python ----------
set "PY_CMD="

python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
    echo [1/4] 检测到系统 Python !PY_VER!
    set "PY_CMD=python"
)

if not defined PY_CMD (
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 (
        echo [1/4] 检测到 Python 3.11 启动器
        set "PY_CMD=py -3.11"
    )
)

if not defined PY_CMD (
    echo [1/4] 未检测到 Python，正在通过 winget 自动安装 Python 3.11...
    winget install --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo [错误] Python 3.11 自动安装失败，请手动到官网下载安装:
        echo   https://www.python.org/downloads/
        echo   安装时务必勾选 "Add python.exe to PATH"
        pause
        exit /b 1
    )
    REM winget 安装后 PATH 不会立即刷新，直接使用常见安装路径
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%PATH%"
    python --version >nul 2>&1
    if not errorlevel 1 (
        echo [1/4] Python 3.11 安装成功
        set "PY_CMD=python"
    ) else (
        echo [错误] Python 安装路径未找到，请重新打开本脚本或重启电脑后重试
        pause
        exit /b 1
    )
)

REM 检查 Python 版本 >= 3.9
for /f "tokens=2 delims= " %%v in ('!PY_CMD! --version 2^>^&1') do set "PY_VER=%%v"
echo    Python 版本: !PY_VER!

REM ---------- 2. 创建虚拟环境 ----------
echo.
echo [2/4] 创建项目虚拟环境 .venv ...
if not exist ".venv\Scripts\python.exe" (
    !PY_CMD! -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo   虚拟环境创建完成
) else (
    echo   虚拟环境已存在，跳过创建
)

set "VPY=.venv\Scripts\python.exe"
set "VPIP=.venv\Scripts\python.exe -m pip"

REM ---------- 3. 安装依赖 ----------
echo.
echo [3/4] 升级 pip 并安装依赖（国内镜像加速，首次安装需下载数 GB 文件，请耐心等待）...

%VPIP% install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1

REM 先安装 CUDA 版 PyTorch（官方源，约 2.5GB）
echo   - 安装 PyTorch (CUDA 12.1) ...
%VPIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo   [警告] CUDA 版 PyTorch 安装失败，尝试从镜像源安装通用版本...
    %VPIP% install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] PyTorch 安装失败，请检查网络后重试
        pause
        exit /b 1
    )
)

REM 安装其余依赖（镜像源）
echo   - 安装其余依赖 ...
%VPIP% install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络后重试
    pause
    exit /b 1
)

REM ---------- 4. 验证 ----------
echo.
echo [4/4] 验证安装 ...
%VPY% -c "import torch, PySide6, cv2, ultralytics; print('  PyTorch:', torch.__version__, '| CUDA 可用:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到'); print('  全部依赖安装成功')"
if errorlevel 1 (
    echo [错误] 环境验证失败
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo   安装完成！双击 start.bat 即可启动工具
echo   （首次使用 AI 辅助分割时会自动下载 SAM 模型）
echo ==========================================================
pause
