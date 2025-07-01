@echo off
chcp 936 >nul
REM �л��� GBK ���루���� Windows Ĭ�ϣ�

REM ���ȼ���Ƿ��Ѵ������⻷��
if exist ".venv_rag\Scripts\activate.bat" (
    call .venv_rag\Scripts\activate.bat
    echo �Ѽ����������⻷��
    goto AFTER_VENV
)

REM ===== ֻ����Ҫ�������⻷��ʱ�ż��Python�汾 =====
REM ��������� Python �汾��Χ��3.10 ~ 3.12��
set "PYTHON_MIN_MAJOR=3"
set "PYTHON_MIN_MINOR=10"
set "PYTHON_MAX_MAJOR=3"
set "PYTHON_MAX_MINOR=12"

echo ���ڼ��ϵͳPython�汾�Ƿ����Ҫ��(3.10-3.12)...

REM ���Ĭ��Python�汾�Ƿ����Ҫ��
set "PYTHON_EXE="
set "PYTHON_VERSION="

REM ��ȡĬ��Python�汾
for /f "tokens=1,2 delims=." %%a in ('python --version 2^>^&1 ^| findstr /r /i "^Python [0-9][0-9]*\.[0-9][0-9]*"') do (
    for /f %%m in ("%%a") do set "MAJOR=%%m"
    set "MINOR=%%b"
    
    REM ����Ƿ� Python 3.10 ~ 3.12
    if "!MAJOR!"=="3" (
        if !MINOR! GEQ %PYTHON_MIN_MINOR% (
            if !MINOR! LEQ %PYTHON_MAX_MINOR% (
                set "PYTHON_EXE=python"
                set "PYTHON_VERSION=3.!MINOR!"
            )
        )
    )
)

REM ���Ĭ��Python������Ҫ������������Python�汾
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

REM ����Ƿ��ҵ�����Ҫ��� Python
if not defined PYTHON_EXE (
    echo ����δ�ҵ� Python 3.10 ~ 3.12 �汾
    echo.
    echo ��ǰϵͳPython�汾: 
    python --version 2>&1
    echo.
    echo ��ִ�����²���֮һ��
    echo 1. �� Python ���� ^(https://www.python.org/downloads/ ^)���ز���װ Python 3.10 ~ 3.12
    echo 2. �����޸ı��ű��е� PYTHON_MIN_MINOR �� PYTHON_MAX_MINOR ����
    echo.
    pause
    exit /b 1
)

echo ����ʹ�� Python !PYTHON_VERSION! �������⻷��...
%PYTHON_EXE% -m venv .venv_rag
call .venv_rag\Scripts\activate.bat

:AFTER_VENV
REM �����￪ʼ�����в����������⻷���н���

REM ��鲢��װrequirements.txt�е�����
if exist "requirements.txt" (
    echo ���ڼ�鲢��װ������...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    echo δ�ҵ� requirements.txt �ļ�
)

REM ���Ҳ��������°汾��RAG�ű�
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
    echo �������� !script!...
    python "!script!"
) else (
    if exist "RAG.py" (
        python RAG.py
    ) else (
        echo δ�ҵ� RAG �ű�
        pause
    )
)
endlocal