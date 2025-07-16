#!/bin/bash

# RAG 启动脚本 for Debian-based Linux

# 解析命令行参数
LOCAL_MODE=false
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--local)
            LOCAL_MODE=true
            shift
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "错误: 未知参数 $1"
            SHOW_HELP=true
            shift
            ;;
    esac
done

# 显示帮助信息
if [ "$SHOW_HELP" = true ]; then
    echo "RAG 启动脚本使用说明"
    echo
    echo "用法: ./RAG.sh [选项]"
    echo
    echo "选项:"
    echo "  -l, --local     启用本地模式"
    echo "  -h, --help      显示此帮助信息"
    echo
    echo "示例:"
    echo "  ./RAG.sh              # 常规启动"
    echo "  ./RAG.sh -l           # 本地模式启动"
    echo "  ./RAG.sh --local      # 本地模式启动"
    echo
    exit 0
fi

# 主程序开始
echo "RAG 启动脚本"
echo "============"

# 检查本地模式
if [ "$LOCAL_MODE" = true ]; then
    echo "正在以本地模式启动..."
    export RAG_LOCAL_MODE=true
else
    echo "正在以标准模式启动..."
    export RAG_LOCAL_MODE=false
fi

# 检查是否存在虚拟环境
if [ -d ".venv_rag/bin" ]; then
    source .venv_rag/bin/activate
    echo "已激活现有虚拟环境"
else
    # 检查Python版本
    echo "正在检查系统Python版本是否符合要求(3.10-3.12)..."
    
    PYTHON_EXE=""
    PYTHON_VERSION=""
    LATEST_MINOR=0
    
    # 检查python命令
    if command -v python &> /dev/null; then
        version=$(python --version 2>&1 | awk '{print $2}')
        IFS=. read -r major minor patch <<< "$version"
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
            if [ "$minor" -gt "$LATEST_MINOR" ]; then
                LATEST_MINOR=$minor
                PYTHON_EXE="python"
                PYTHON_VERSION="$major.$minor"
            fi
        fi
    fi
    
    # 检查python3命令
    if command -v python3 &> /dev/null; then
        version=$(python3 --version 2>&1 | awk '{print $2}')
        IFS=. read -r major minor patch <<< "$version"
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
            if [ "$minor" -gt "$LATEST_MINOR" ]; then
                LATEST_MINOR=$minor
                PYTHON_EXE="python3"
                PYTHON_VERSION="$major.$minor"
            fi
        fi
    fi
    
    if [ -z "$PYTHON_EXE" ]; then
        echo "错误：未找到 Python 3.10 ~ 3.12 版本"
        echo
        echo "当前系统Python版本:"
        python --version 2>/dev/null || echo "未找到Python"
        python3 --version 2>/dev/null || echo "未找到Python3"
        echo
        echo "请执行以下操作之一："
        echo "1. 使用 apt-get install python3.10 或类似命令安装 Python 3.10 ~ 3.12"
        echo "2. 或者修改本脚本中的版本检查逻辑"
        echo
        exit 1
    fi
    
    echo "正在使用 Python $PYTHON_VERSION 创建虚拟环境..."
    $PYTHON_EXE -m venv .venv_rag
    source .venv_rag/bin/activate
fi

# 检查并安装依赖
if [ -f "requirements.txt" ]; then
    echo "正在检查并安装依赖库..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    echo "未找到 requirements.txt 文件"
fi

# 查找并运行最新版本的RAG脚本
MAX_VERSION=0
SCRIPT_TO_RUN=""

for file in RAGv*.py; do
    if [[ -f "$file" ]]; then
        basename="${file%.*}"
        version_part="${basename:4}"
        if [[ "$version_part" =~ ^[0-9]+$ ]] && [ "$version_part" -gt "$MAX_VERSION" ]; then
            MAX_VERSION="$version_part"
            SCRIPT_TO_RUN="$file"
        fi
    fi
done

# 准备Python脚本参数
PYTHON_ARGS=""
if [ "$LOCAL_MODE" = true ]; then
    PYTHON_ARGS="--local"
fi

# 运行脚本
if [ "$MAX_VERSION" -gt 0 ]; then
    echo "正在运行 $SCRIPT_TO_RUN..."
    if [ -n "$PYTHON_ARGS" ]; then
        python "$SCRIPT_TO_RUN" $PYTHON_ARGS
    else
        python "$SCRIPT_TO_RUN"
    fi
elif [ -f "RAG.py" ]; then
    echo "正在运行 RAG.py..."
    if [ -n "$PYTHON_ARGS" ]; then
        python RAG.py $PYTHON_ARGS
    else
        python RAG.py
    fi
else
    echo "未找到 RAG 脚本"
    read -p "按任意键继续..." -n1 -s
    echo
fi