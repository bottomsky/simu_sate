# C++ 开发基础镜像构建脚本
# 用于构建包含常用 C++ 开发依赖的基础镜像

param(
    [switch]$NoCache,
    [switch]$Push,
    [string]$Tag = "cpp-dev-base:latest",
    [switch]$Help
)

# 颜色输出函数
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# 显示使用说明
function Show-Usage {
    Write-Host "C++ 开发基础镜像构建脚本" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "用法: .\build-base.ps1 [参数]"
    Write-Host ""
    Write-Host "参数:"
    Write-Host "  -NoCache      不使用 Docker 缓存重新构建"
    Write-Host "  -Push         构建完成后推送到镜像仓库"
    Write-Host "  -Tag          指定镜像标签 (默认: cpp-dev-base:latest)"
    Write-Host "  -Help         显示此帮助信息"
    Write-Host ""
    Write-Host "示例:"
    Write-Host "  .\build-base.ps1                           # 基本构建"
    Write-Host "  .\build-base.ps1 -NoCache                  # 无缓存构建"
    Write-Host "  .\build-base.ps1 -Tag my-cpp-base:v1.0    # 自定义标签"
    Write-Host "  .\build-base.ps1 -Push                     # 构建并推送"
}

# 检查 Docker 是否可用
function Test-Docker {
    try {
        $null = Get-Command docker -ErrorAction Stop
        $null = docker info 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker 服务未运行"
        }
        Write-Success "Docker 环境检查通过"
        return $true
    }
    catch {
        Write-Error "Docker 未安装或服务未运行: $($_.Exception.Message)"
        return $false
    }
}

# 构建基础镜像
function Build-BaseImage {
    Write-Info "开始构建 C++ 开发基础镜像..."
    Write-Info "镜像标签: $Tag"
    
    # 确保在项目根目录
    if (-not (Test-Path "CMakeLists.txt")) {
        Write-Error "请在项目根目录运行此脚本"
        return $false
    }
    
    try {
        # 构建参数
        $buildArgs = @(
            "build",
            "-f", "docker/Dockerfile.base",
            "-t", $Tag
        )
        
        if ($NoCache) {
            $buildArgs += "--no-cache"
            Write-Info "使用 --no-cache 选项"
        }
        
        $buildArgs += "."
        
        Write-Info "执行构建命令: docker $($buildArgs -join ' ')"
        
        # 执行构建
        $buildResult = & docker @buildArgs 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "基础镜像构建失败"
            Write-Host $buildResult -ForegroundColor Red
            return $false
        }
        
        Write-Success "基础镜像构建完成"
        return $true
    }
    catch {
        Write-Error "构建过程中出现异常: $($_.Exception.Message)"
        return $false
    }
}

# 验证构建结果
function Test-BaseImage {
    Write-Info "验证基础镜像..."
    
    try {
        # 运行容器并检查安装的工具
        $verifyScript = @"
echo '=== 验证开发工具 ==='
gcc --version
g++ --version
cmake --version
git --version
pkg-config --version
echo '=== 验证 Eigen 库 ==='
find /usr -name "Eigen" -type d 2>/dev/null | head -3
echo '=== 验证完成 ==='
"@
        
        $result = docker run --rm $Tag sh -c $verifyScript 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "基础镜像验证完成"
            Write-Host $result -ForegroundColor Gray
        } else {
            Write-Warning "验证过程中出现问题，但镜像可能仍然可用"
        }
    }
    catch {
        Write-Warning "验证过程中出现异常: $($_.Exception.Message)"
    }
}

# 推送镜像
function Push-Image {
    if ($Push) {
        Write-Info "推送镜像到仓库..."
        try {
            docker push $Tag
            if ($LASTEXITCODE -eq 0) {
                Write-Success "镜像推送完成"
            } else {
                Write-Error "镜像推送失败"
            }
        }
        catch {
            Write-Error "推送过程中出现异常: $($_.Exception.Message)"
        }
    }
}

# 显示镜像信息
function Show-ImageInfo {
    Write-Info "镜像信息:"
    try {
        $imageInfo = docker images $Tag --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        Write-Host $imageInfo -ForegroundColor Gray
    }
    catch {
        Write-Warning "无法获取镜像信息"
    }
}

# 主函数
function Main {
    if ($Help) {
        Show-Usage
        return
    }
    
    Write-Info "开始 C++ 开发基础镜像构建流程..."
    
    # 检查 Docker 环境
    if (-not (Test-Docker)) {
        return
    }
    
    # 构建基础镜像
    if (-not (Build-BaseImage)) {
        return
    }
    
    # 验证镜像
    Test-BaseImage
    
    # 推送镜像（如果需要）
    Push-Image
    
    # 显示镜像信息
    Show-ImageInfo
    
    Write-Success "C++ 开发基础镜像构建流程完成!"
    Write-Info "镜像标签: $Tag"
    Write-Info "使用方法: 在其他 Dockerfile 中使用 'FROM $Tag'"
}

# 运行主函数
Main