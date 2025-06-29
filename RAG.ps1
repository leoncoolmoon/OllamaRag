# 设置需要的 Python 版本
$requiredVersion = "3.10"
$pythonExe = "python$requiredVersion"

# 检查是否存在指定版本
if (-not (Get-Command $pythonExe -ErrorAction SilentlyContinue)) {
    Write-Host "错误：未找到 Python $requiredVersion" -ForegroundColor Red
    Write-Host "已安装的 Python 版本："
    Get-Command python* | Select-Object Name, Source | Format-Table -AutoSize
    
    # 提供安装建议
    Write-Host "`n解决方案：" -ForegroundColor Yellow
    Write-Host "1. 从 Microsoft Store 安装: 搜索 'Python $requiredVersion'"
    Write-Host "2. 或从 python.org 下载: https://www.python.org/downloads/"
    exit 1
}

# 创建/激活虚拟环境
$venvPath = ".venv_rag"
if (Test-Path "$venvPath/Scripts/activate.ps1") {
    & "$venvPath/Scripts/activate.ps1"
} else {
    Write-Host "正在创建虚拟环境（使用 Python $requiredVersion）..."
    & $pythonExe -m venv $venvPath
    & "$venvPath/Scripts/activate.ps1"
}

# 安装依赖
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
}

# 运行 RAG 脚本
$script = Get-ChildItem RAGv*.py | Sort-Object { [regex]::Match($_.Name, 'RAGv(\d+)').Groups[1].Value } -Descending | Select-Object -First 1
if ($script) {
    Write-Host "执行最新版本脚本: $($script.Name)"
    & python $script.Name
} elseif (Test-Path "RAG.py") {
    & python RAG.py
} else {
    Write-Host "错误：未找到 RAG 脚本" -ForegroundColor Red
    exit 1
}