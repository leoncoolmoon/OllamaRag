@echo off
REM 设置Python版本变量
set "PYTHON_VERSION=3.10"
set "PYTHON_EXE=python%PYTHON_VERSION%"

REM 检查指定的Python版本是否可用
where %PYTHON_EXE% >nul 2>&1
if %ERRORLEVEL% neq 0 (
        echo 错误：未找到 Python %PYTHON_VERSION%
            echo.
                echo 请执行以下操作之一：
                    echo 1. 从 Python 官网(https://www.python.org/downloads/)下载并安装 Python %PYTHON_VERSION%
                        echo 2. 或者修改本脚本中的 PYTHON_VERSION 变量为已安装的 Python 版本
                            echo.
                                pause
                                    exit /b 1
)

REM 激活虚拟环境
if exist ".venv_rag\Scripts\activate.bat" (
        call .venv_rag\Scripts\activate.bat) else (
                echo 虚拟环境 .venv_rag 不存在，正在使用 Python %PYTHON_VERSION% 创建...
                    %PYTHON_EXE% -m venv .venv_rag    call .venv_rag\Scripts\activate.bat)

                    REM 检查并安装requirements.txt中的依赖
                    if exist "requirements.txt" (
                            echo 正在检查并安装依赖库...
                                pip install --upgrade pip    pip install -r requirements.txt) else (
                                        echo 未找到 requirements.txt 文件
                                )

                                REM 查找所有 RAGv*.py 脚本，提取数字并找出最大的一个
                                setlocal enabledelayedexpansionset "max=0"
                                for %%f in (RAGv*.py) do (
                                        set "name=%%~nf"
                                            REM 提取数字部分
                                                for /f "tokens=2 delims=v" %%a in ("!name!") do (
                                                            set "num=%%a"
                                                                    REM 去除可能的非数字后缀
                                                                            for /f "delims=.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" %%b in ("!num!") do (
                                                                                            if %%b gtr !max! set "max=%%b"
                                                                            )
                                                )
                                )

                                REM 构造最大版本号的脚本名并运行
                                if not "!max!"=="0" (
                                        set "script=RAGv!max!.py"
                                            echo Running !script!...
                                                python "!script!"
                                ) else (
                                        echo 未找到符合条件的脚本
                                            if exist "RAG.py" (
                                                        python RAG.py    ) else (
                                                                    echo 未找到 RAG.py 脚本，提取数字并找出最大的一个        pause    )
                                )
                                endlocal
                                  