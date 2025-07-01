# PowerShell equivalent of the batch script
# Encoding is UTF-8 by default in PowerShell, no need for chcp

function CheckPythonVersion {
    param(
        [int]$minMajor = 3,
        [int]$minMinor = 10,
        [int]$maxMajor = 3,
        [int]$maxMinor = 12
    )

    Write-Host "正在检查系统Python版本是否符合要求(3.10-3.12)..."

    $pythonExe = $null
    $pythonVersion = $null
    $latestMinor = 0

    # Check Python installations
    $pythonPaths = @(Get-Command python -All -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
    $pythonPaths += @(Get-Command python3 -All -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
    $pythonPaths = $pythonPaths | Sort-Object -Unique

    foreach ($path in $pythonPaths) {
        try {
            $versionOutput = & $path --version 2>&1
            if ($versionOutput -match 'Python (\d+)\.(\d+)') {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]

                if ($major -eq $minMajor -and $minor -ge $minMinor -and $minor -le $maxMinor) {
                    if ($minor -gt $latestMinor) {
                        $latestMinor = $minor
                        $pythonExe = $path
                        $pythonVersion = "$major.$minor"
                    }
                }
            }
        } catch {
            continue
        }
    }

    if (-not $pythonExe) {
        Write-Host "错误：未找到 Python 3.10 ~ 3.12 版本"
        Write-Host ""
        Write-Host "当前系统Python版本:"
        try { python --version 2>&1 } catch { Write-Host "未找到Python" }
        Write-Host ""
        Write-Host "请执行以下操作之一："
        Write-Host "1. 从 Python 官网 (https://www.python.org/downloads/) 下载并安装 Python 3.10 ~ 3.12"
        Write-Host "2. 或者修改本脚本中的 PYTHON_MIN_MINOR 和 PYTHON_MAX_MINOR 变量"
        Write-Host ""
        Pause
        exit 1
    }

    return @{
        Exe = $pythonExe
        Version = $pythonVersion
    }
}

# Main script execution
if (Test-Path ".venv_rag\Scripts\Activate.ps1") {
    & .venv_rag\Scripts\Activate.ps1
    Write-Host "已激活现有虚拟环境"
} else {
    $pythonInfo = CheckPythonVersion
    Write-Host "正在使用 Python $($pythonInfo.Version) 创建虚拟环境..."
    & $pythonInfo.Exe -m venv .venv_rag
    & .venv_rag\Scripts\Activate.ps1
}

# From here, all operations are in the virtual environment

# Check and install requirements.txt
if (Test-Path "requirements.txt") {
    Write-Host "正在检查并安装依赖库..."
    & python -m pip install --upgrade pip
    & python -m pip install -r requirements.txt
} else {
    Write-Host "未找到 requirements.txt 文件"
}

# Find and run the latest version of RAG script
$maxVersion = 0
$scriptToRun = $null

Get-ChildItem -Filter "RAGv*.py" | ForEach-Object {
    if ($_.Name -match 'RAGv(\d+)') {
        $version = [int]$Matches[1]
        if ($version -gt $maxVersion) {
            $maxVersion = $version
            $scriptToRun = $_.FullName
        }
    }
}

if ($maxVersion -gt 0) {
    Write-Host "正在运行 $($scriptToRun)..."
    & python $scriptToRun
} elseif (Test-Path "RAG.py") {
    & python RAG.py
} else {
    Write-Host "未找到 RAG 脚本"
    Pause
}