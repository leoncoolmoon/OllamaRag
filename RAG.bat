@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: 解析命令行参数
set "LOCAL_MODE=false"
set "SHOW_HELP=false"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="-l" (
    set "LOCAL_MODE=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--local" (
    set "LOCAL_MODE=true"
    shift
    goto :parse_args
)
if /i "%~1"=="-h" (
    set "SHOW_HELP=true"
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    set "SHOW_HELP=true"
    shift
    goto :parse_args
)
echo 错误: 未知参数 %~1
goto :show_help
shift
goto :parse_args

:args_done

:: 显示帮助信息
if "%SHOW_HELP%"=="true" goto :show_help

:: 主程序开始
echo RAG 启动脚本
echo ============

:: 检查本地模式
if "%LOCAL_MODE%"=="true" (
    echo 正在以本地模式启动...
    set "RAG_LOCAL_MODE=true"
) else (
    echo 正在以标准模式启动...
    set "RAG_LOCAL_MODE=false"
)

:: 检查是否存在虚拟环境
if exist ".venv_rag\Scripts\activate.bat" (
    call .venv_rag\Scripts\activate.bat
    echo 已激活现有虚拟环境
) else (
    call :check_python_version
    if !errorlevel! neq 0 exit /b 1
    echo 正在使用 Python !PYTHON_VERSION! 创建虚拟环境...
    "!PYTHON_EXE!" -m venv .venv_rag
    call .venv_rag\Scripts\activate.bat
)

:: 检查并安装依赖
if exist "requirements.txt" (
    echo 正在检查并安装依赖库...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    echo 未找到 requirements.txt 文件
)

:: 查找并运行最新版本的RAG脚本
set "MAX_VERSION=0"
set "SCRIPT_TO_RUN="

for %%f in (RAGv*.py) do (
    set "filename=%%f"
    for /f "tokens=1 delims=." %%a in ("!filename!") do (
        set "basename=%%a"
        set "version_part=!basename:~4!"
        if "!version_part!" gtr "!MAX_VERSION!" (
            set "MAX_VERSION=!version_part!"
            set "SCRIPT_TO_RUN=%%f"
        )
    )
)

:: 准备Python脚本参数
set "PYTHON_ARGS="
if "%LOCAL_MODE%"=="true" (
    set "PYTHON_ARGS=--local"
)

:: 运行脚本
if !MAX_VERSION! gtr 0 (
    echo 正在运行 !SCRIPT_TO_RUN!...
    if defined PYTHON_ARGS (
        python "!SCRIPT_TO_RUN!" !PYTHON_ARGS!
    ) else (
        python "!SCRIPT_TO_RUN!"
    )
) else if exist "RAG.py" (
    echo 正在运行 RAG.py...
    if defined PYTHON_ARGS (
        python RAG.py !PYTHON_ARGS!
    ) else (
        python RAG.py
    )
) else (
    echo 未找到 RAG 脚本
    pause
)

goto :eof

:check_python_version
echo 正在检查系统Python版本是否符合要求(3.10-3.12)...

set "PYTHON_EXE="
set "PYTHON_VERSION="
set "LATEST_MINOR=0"

:: 检查python命令
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do (
    call :parse_version "%%v"
)

:: 检查python3命令
for /f "tokens=2 delims= " %%v in ('python3 --version 2^>^&1') do (
    call :parse_version "%%v"
)

if not defined PYTHON_EXE (
    echo 错误：未找到 Python 3.10 ~ 3.12 版本
    echo.
    echo 当前系统Python版本:
    python --version 2>nul || echo 未找到Python
    echo.
    echo 请执行以下操作之一：
    echo 1. 从 Python 官网 ^(https://www.python.org/downloads/^) 下载并安装 Python 3.10 ~ 3.12
    echo 2. 或者修改本脚本中的版本检查逻辑
    echo.
    pause
    exit /b 1
)

exit /b 0

:parse_version
set "version_str=%~1"
for /f "tokens=1,2 delims=." %%a in ("%version_str%") do (
    set "major=%%a"
    set "minor=%%b"
)

if "%major%"=="3" (
    if %minor% geq 10 if %minor% leq 12 (
        if %minor% gtr %LATEST_MINOR% (
            set "LATEST_MINOR=%minor%"
            set "PYTHON_EXE=python"
            set "PYTHON_VERSION=%major%.%minor%"
        )
    )
)
exit /b 0

:show_help
echo RAG 启动脚本使用说明
echo.
echo 用法: RAG.bat [选项]
echo.
echo 选项:
echo   -l, --local     启用本地模式
echo   -h, --help      显示此帮助信息
echo.
echo 示例:
echo   RAG.bat              # 常规启动
echo   RAG.bat -l           # 本地模式启动
echo   RAG.bat --local      # 本地模式启动
echo.
exit /b 0