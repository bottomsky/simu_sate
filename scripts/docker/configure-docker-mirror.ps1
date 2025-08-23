<#
.SYNOPSIS
    配置 Docker 镜像源以解决网络连接问题

.DESCRIPTION
    此脚本帮助配置 Docker Desktop 使用国内镜像源，解决拉取镜像时的网络连接问题。
    支持自动检测 Docker Desktop 配置路径并应用镜像源配置。

.PARAMETER Apply
    是否自动应用配置到 Docker Desktop

.EXAMPLE
    .\configure-docker-mirror.ps1
    显示配置说明

.EXAMPLE
    .\configure-docker-mirror.ps1 -Apply
    自动应用镜像源配置
#>

param(
    [switch]$Apply
)

# 颜色输出函数
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# 检查 Docker Desktop 是否运行
function Test-DockerDesktop {
    try {
        $dockerProcess = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
        return $dockerProcess -ne $null
    }
    catch {
        return $false
    }
}

# 获取 Docker Desktop 配置路径
function Get-DockerConfigPath {
    $possiblePaths = @(
        "$env:APPDATA\Docker\settings.json",
        "$env:USERPROFILE\.docker\daemon.json",
        "C:\ProgramData\Docker\config\daemon.json"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path (Split-Path $path -Parent)) {
            return $path
        }
    }
    
    return $null
}

Write-ColorOutput "=== Docker 镜像源配置工具 ===" "Cyan"
Write-ColorOutput ""

# 检查 Docker 状态
if (-not (Test-DockerDesktop)) {
    Write-ColorOutput "警告: Docker Desktop 未运行，请先启动 Docker Desktop" "Yellow"
}

# 显示当前目录下的配置文件
$currentDaemonJson = Join-Path $PSScriptRoot "daemon.json"
if (Test-Path $currentDaemonJson) {
    Write-ColorOutput "找到镜像源配置文件: $currentDaemonJson" "Green"
    Write-ColorOutput "配置内容:" "White"
    Get-Content $currentDaemonJson | Write-Host
    Write-ColorOutput ""
else {
    Write-ColorOutput "错误: 未找到 daemon.json 配置文件" "Red"
    exit 1
}

if ($Apply) {
    Write-ColorOutput "正在应用镜像源配置..." "Yellow"
    
    # 获取 Docker 配置路径
    $dockerConfigPath = Get-DockerConfigPath
    
    if ($dockerConfigPath) {
        try {
            # 创建配置目录（如果不存在）
            $configDir = Split-Path $dockerConfigPath -Parent
            if (-not (Test-Path $configDir)) {
                New-Item -ItemType Directory -Path $configDir -Force | Out-Null
            }
            
            # 复制配置文件
            Copy-Item $currentDaemonJson $dockerConfigPath -Force
            Write-ColorOutput "配置已应用到: $dockerConfigPath" "Green"
            Write-ColorOutput "请重启 Docker Desktop 以使配置生效" "Yellow"
        }
        catch {
            Write-ColorOutput "应用配置失败: $($_.Exception.Message)" "Red"
            Write-ColorOutput "请手动配置 Docker Desktop 镜像源" "Yellow"
        }
    }
    else {
        Write-ColorOutput "无法自动检测 Docker 配置路径，请手动配置" "Yellow"
    }
}
else {
    Write-ColorOutput "手动配置步骤:" "Cyan"
    Write-ColorOutput "1. 打开 Docker Desktop" "White"
    Write-ColorOutput "2. 点击设置图标 (齿轮图标)" "White"
    Write-ColorOutput "3. 选择 'Docker Engine'" "White"
    Write-ColorOutput "4. 在 JSON 配置中添加以下内容:" "White"
    Write-ColorOutput ""
    Write-ColorOutput '  "registry-mirrors": [' "Gray"
    Write-ColorOutput '    "https://docker.mirrors.ustc.edu.cn",' "Gray"
    Write-ColorOutput '    "https://hub-mirror.c.163.com",' "Gray"
    Write-ColorOutput '    "https://mirror.baidubce.com",' "Gray"
    Write-ColorOutput '    "https://ccr.ccs.tencentyun.com"' "Gray"
    Write-ColorOutput '  ]' "Gray"
    Write-ColorOutput ""
    Write-ColorOutput "5. 点击 'Apply & Restart'" "White"
    Write-ColorOutput ""
    Write-ColorOutput "或者运行: .\configure-docker-mirror.ps1 -Apply" "Green"
}

Write-ColorOutput ""
Write-ColorOutput "配置完成后，可以运行以下命令测试:" "Cyan"
Write-ColorOutput "docker pull alpine:latest" "Gray"
Write-ColorOutput ""
Write-ColorOutput "如果仍有问题，请检查网络连接或尝试其他镜像源" "Yellow"