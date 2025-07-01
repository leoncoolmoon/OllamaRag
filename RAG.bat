@echo off
chcp 936 >nul
REM 切换到 GBK 编码（中文 Windows 默认）

REM 首先检查是否已存在虚拟环境
if exist ".venv_rag\Scripts\activate.bat" (
    call .venv_rag\Scripts\activate.bat
    echo 已激活现有虚拟环境
    goto AFTER_VENV
)

REM ===== 只有需要创建虚拟环境时才检查Python版本 =====
REM 设置允许的 Python 版本范围（3.10 ~ 3.12）
set "PYTHON_MIN_MAJOR=3"
set "PYTHON_MIN_MINOR=10"
set "PYTHON_MAX_MAJOR=3"
set "PYTHON_MAX_MINOR=12"

echo 正在检查系统Python版本是否符合要求(3.10-3.12)...

REM 检查默认Python版本是否符合要求
set "PYTHON_EXE="
set "PYTHON_VERSION="

REM 获取默认Python版本
for /f "tokens=1,2 delims=." %%a in ('python --version 2^>^&1 ^| findstr /r /i "^Python [0-9][0-9]*\.[0-9][0-9]*"') do (
    for /f %%m in ("%%a") do set "MAJOR=%%m"
    set "MINOR=%%b"
    
    REM 检查是否 Python 3.10 ~ 3.12
    if "!MAJOR!"=="3" (
        if !MINOR! GEQ %PYTHON_MIN_MINOR% (
            if !MINOR! LEQ %PYTHON_MAX_MINOR% (
                set "PYTHON_EXE=python"
                set "PYTHON_VERSION=3.!MINOR!"
            )
        )
    )
)

REM 如果默认Python不符合要求，则搜索其他Python版本
if not defined PYTHON_EXE (
    set "LATEST_MINOR=0"
    for /f "tokens=*" %%p in ('where python 2^>nul') do (
        for /f "tokens=1,2 delims=." %%a in ('%%p --version 2^>^&1 ^| findstr /r /i "^Python [0-9][0-9]*\.[0-9][0-9]*"') do (
            for /f %%m in ("%%a") do set "MAJOR=%%m"
            set "MINOR=%%b"

            if "!MAJOR!"=="3" (
                if !MINOR! GEQ %PYTHON_MIN_MINOR% (
                    if !MINOR! LEQ %PYTHON_MAX_MINOR% (
                        if !MINOR! GTR !LATEST_MINOR! (
                            set "LATEST_MINOR=!MINOR!"
                            set "PYTHON_EXE=%%p"
                            set "PYTHON_VERSION=3.!MINOR!"
                        )
                    )
                )
            )
        )
    )
)

REM 检查是否找到符合要求的 Python
if not defined PYTHON_EXE (
    echo 错误：未找到 Python 3.10 ~ 3.12 版本
    echo.
    echo 当前系统Python版本: 
    python --version 2>&1
    echo.
    echo 请执行以下操作之一：
    echo 1. 从 Python 官网 ^(https://www.python.org/downloads/ ^)下载并安装 Python 3.10 ~ 3.12
    echo 2. 或者修改本脚本中的 PYTHON_MIN_MINOR 和 PYTHON_MAX_MINOR 变量
    echo.
    pause
    exit /b 1
)

echo 正在使用 Python !PYTHON_VERSION! 创建虚拟环境...
%PYTHON_EXE% -m venv .venv_rag
call .venv_rag\Scripts\activate.bat

:AFTER_VENV
REM 从这里开始，所有操作都在虚拟环境中进行

REM 检查并安装requirements.txt中的依赖
if exist "requirements.txt" (
    echo 正在检查并安装依赖库...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    echo 未找到 requirements.txt 文件
)

REM 查找并运行最新版本的RAG脚本
setlocal enabledelayedexpansion
set "max=0"
for %%f in (RAGv*.py) do (
    set "name=%%~nf"
    for /f "tokens=2 delims=v" %%a in ("!name!") do (
        set "num=%%a"
        for /f "delims=.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" %%b in ("!num!") do (
            if %%b gtr !max! set "max=%%b"
        )
    )
)

if not "!max!"=="0" (
    set "script=RAGv!max!.py"
    echo 正在运行 !script!...
    python "!script!"
) else (
    if exist "RAG.py" (
        python RAG.py
    ) else (
        echo 未找到 RAG 脚本
        pause
    )
)
endlocal