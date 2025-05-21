@echo off
REM 激活虚拟环境
call .venv_rag\Scripts\activate.bat

REM 查找所有 RAGv*.py 脚本，提取数字并找出最大的一个
setlocal enabledelayedexpansion
set "max=0"
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
    python RAG.py

)
endlocal
