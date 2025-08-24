# Alpine Linux 构建脚本 (PowerShell 版本)
# 使用 prantlf/alpine-make-gcc:latest 基础镜像构建 J2 Orbit Propagator

param(
    [switch]$NoCleanup,
    [switch]$NoExtract,
    [switch]$WithMount,
    [string]$MountPath = "build/Release",
    [switch]$BuildBaseOnly,
    [switch]$SkipBase,
    [switch]$Help
)

# 颜色输出函数
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
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
    Write-Host "Alpine Linux 构建脚本使用说明:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "用法: .\build.ps1 [参数]"
    Write-Host ""
    Write-Host "参数:"
    Write-Host "  -NoCleanup     跳过清理步骤"
    Write-Host "  -NoExtract     跳过提取构建产物"
    Write-Host "  -WithMount     使用挂载方式直接输出到主机目录"
    Write-Host "  -MountPath     指定挂载的主机路径 (默认: build/Release)"
    Write-Host "  -BuildBaseOnly 仅构建基础镜像"
    Write-Host "  -SkipBase      跳过基础镜像构建（假设已存在）"
    Write-Host "  -Help          显示此帮助信息"
    Write-Host ""
    Write-Host "路径说明:"
    Write-Host "  - 在 docker 目录运行时，相对路径相对于 docker 目录解析"
    Write-Host "  - 默认 'build/Release' 会解析为项目根目录下的 build/Release 目录"
    Write-Host "  - 支持绝对路径和相对路径"
    Write-Host ""
    Write-Host "示例:"
    Write-Host "  .\build.ps1                           # 完整构建流程（包含基础镜像）"
    Write-Host "  .\build.ps1 -BuildBaseOnly           # 仅构建基础镜像"
    Write-Host "  .\build.ps1 -SkipBase                # 跳过基础镜像构建"
    Write-Host "  .\build.ps1 -NoCleanup               # 构建但不清理旧镜像"
    Write-Host "  .\build.ps1 -WithMount               # 使用挂载方式输出到项目根目录/build/Release"
    Write-Host "  .\build.ps1 -WithMount -MountPath D:\output  # 挂载到指定目录"
    Write-Host "  .\build.ps1 -WithMount -MountPath ../build/Release     # 挂载到上级目录的 build/Release"
    Write-Host ""
    Write-Host "基础镜像说明:"
    Write-Host "  基础镜像包含: cmake, git, pkgconfig, eigen-dev, linux-headers"
    Write-Host "  基础镜像名称: cpp-dev-base:latest"
    Write-Host ""
    Write-Host "挂载模式说明:"
    Write-Host "  使用 -WithMount 参数时，构建产物将直接输出到主机指定目录"
    Write-Host "  容器内路径: /output"
    Write-Host "  默认主机路径: ../build/Release"
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
        
        # 检查网络连接
        Write-Info "检查 Docker 网络连接..."
        Test-DockerNetworkConnection
        
        return $true
    }
    catch {
        Write-Error "Docker 未安装或服务未运行: $($_.Exception.Message)"
        return $false
    }
}

# 检查 Docker 网络连接
function Test-DockerNetworkConnection {
    Write-Info "测试 Docker Hub 连接..."
    
    try {
        # 尝试拉取一个小镜像来测试网络
        $testResult = docker pull hello-world:latest 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Docker Hub 连接正常"
            # 清理测试镜像
            docker rmi hello-world:latest 2>$null | Out-Null
        } else {
            Write-Warning "Docker Hub 连接可能有问题"
            Write-Info "建议配置镜像源以解决网络问题"
            Write-Info "运行: .\configure-docker-mirror.ps1 -Apply"
            
            # 检查是否有镜像源配置文件
            if (Test-Path "daemon.json") {
                Write-Info "发现镜像源配置文件: daemon.json"
                Write-Info "请运行 configure-docker-mirror.ps1 脚本配置镜像源"
            }
        }
    }
    catch {
        Write-Warning "网络连接检查失败: $($_.Exception.Message)"
        Write-Info "将继续构建流程，如遇网络问题请配置镜像源"
    }
}

# 带重试的 Docker 构建函数
function Invoke-DockerBuildWithRetry {
    param(
        [string]$DockerfilePath,
        [string]$Tag,
        [string]$Context = ".",
        [int]$MaxRetries = 3,
        [int]$RetryDelay = 10
    )
    
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        Write-Info "构建尝试 $attempt/$MaxRetries..."
        
        $buildResult = docker build -f $DockerfilePath -t $Tag $Context 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "构建成功 (尝试 $attempt/$MaxRetries)"
            return $true
        }
        
        Write-Warning "构建失败 (尝试 $attempt/$MaxRetries)"
        
        # 检查是否是网络相关错误
        $networkErrors = @(
            "failed to do request",
            "dial tcp",
            "connection refused",
            "timeout",
            "registry-1.docker.io"
        )
        
        $isNetworkError = $false
        foreach ($errorMsg in $networkErrors) {
            if ($buildResult -match $errorMsg) {
                $isNetworkError = $true
                break
            }
        }
        
        if ($isNetworkError) {
            Write-Warning "检测到网络相关错误"
            if (Test-Path "configure-docker-mirror.ps1") {
                Write-Info "建议配置 Docker 镜像源: .\configure-docker-mirror.ps1 -Apply"
            }
        }
        
        if ($attempt -lt $MaxRetries) {
            Write-Info "等待 $RetryDelay 秒后重试..."
            Start-Sleep -Seconds $RetryDelay
        } else {
            Write-Error "所有构建尝试都失败了"
            Write-Host $buildResult -ForegroundColor Red
        }
    }
    
    return $false
}

# 清理旧的镜像和容器
function Clear-OldArtifacts {
    Write-Info "清理旧的 Docker 镜像和容器..."
    
    try {
        # 停止并删除相关容器
        $containers = docker ps -a --filter "ancestor=j2-orbit-propagator-alpine" --format "{{.ID}}" 2>$null
        if ($containers) {
            docker rm -f $containers 2>$null | Out-Null
        }
        
        # 删除旧镜像
        $images = docker images --filter "reference=j2-orbit-propagator-alpine" --format "{{.ID}}" 2>$null
        if ($images) {
            docker rmi $images 2>$null | Out-Null
        }
        
        Write-Success "清理完成"
    }
    catch {
        Write-Warning "清理过程中出现问题: $($_.Exception.Message)"
    }
}

# 构建基础镜像
function Build-BaseImage {
    Write-Info "构建 C++ 开发基础镜像..."
    
    # 检查基础镜像 Dockerfile 是否存在
    if (-not (Test-Path "Dockerfile.base")) {
        Write-Error "基础镜像 Dockerfile 不存在: Dockerfile.base"
        return $false
    }
    
    try {
        # 使用重试机制构建基础镜像
        Write-Info "开始构建基础镜像 cpp-dev-base:latest..."
        
        if (-not (Invoke-DockerBuildWithRetry -DockerfilePath "Dockerfile.base" -Tag "cpp-dev-base:latest" -Context "..")) {
            Write-Error "基础镜像构建失败"
            return $false
        }
        
        Write-Success "基础镜像构建完成: cpp-dev-base:latest"
        
        # 验证基础镜像
        Write-Info "验证基础镜像..."
        $verifyResult = docker run --rm cpp-dev-base:latest sh -c "gcc --version && cmake --version && git --version" 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "基础镜像验证通过"
        } else {
            Write-Warning "基础镜像验证失败，但构建可能仍然成功"
        }
        
        return $true
    }
    catch {
        Write-Error "基础镜像构建过程中出现异常: $($_.Exception.Message)"
        return $false
    }
}

# 检查基础镜像是否存在
function Test-BaseImage {
    try {
        $result = docker images cpp-dev-base:latest --format "{{.ID}}" 2>$null
        return ($result -and $result.Trim() -ne "")
    }
    catch {
        return $false
    }
}

# 构建镜像
function Build-Image {
    Write-Info "开始构建 Alpine 版本的 J2 Orbit Propagator..."
    
    # 确保在项目根目录或 docker 目录
    if (-not (Test-Path "CMakeLists.txt") -and -not (Test-Path "../CMakeLists.txt")) {
        Write-Error "请在项目根目录或 docker 目录运行此脚本"
        return $false
    }
    
    # 如果在 docker 目录，切换到项目根目录进行构建
    $originalLocation = Get-Location
    if (Test-Path "../CMakeLists.txt") {
        Set-Location ".."
    }
    
    try {
        # 使用重试机制构建镜像
        $dockerfilePath = if (Test-Path "docker/Dockerfile") { "docker/Dockerfile" } else { "Dockerfile" }
        
        if (-not (Invoke-DockerBuildWithRetry -DockerfilePath $dockerfilePath -Tag "j2-orbit-propagator-alpine:latest" -Context ".")) {
            Write-Error "Docker 构建失败"
            return $false
        }
        
        Write-Success "镜像构建完成"
        
        # 恢复原始位置
        if ($originalLocation) {
            Set-Location $originalLocation
        }
        
        return $true
    }
    catch {
        Write-Error "构建过程中出现异常: $($_.Exception.Message)"
        
        # 恢复原始位置
        if ($originalLocation) {
            Set-Location $originalLocation
        }
        
        return $false
    }
}

# 使用挂载方式构建并直接输出到主机目录
function Build-WithMount {
    param([string]$HostPath)
    
    Write-Info "使用挂载方式构建，输出到: $HostPath"
    
    # 确保在项目根目录或 docker 目录
    if (-not (Test-Path "CMakeLists.txt") -and -not (Test-Path "../CMakeLists.txt")) {
        Write-Error "请在项目根目录或 docker 目录运行此脚本"
        return $false
    }
    
    # 在切换目录之前先解析绝对路径
    $originalLocation = Get-Location
    $absolutePath = $null
    
    # 如果在 docker 目录，需要相对于项目根目录解析路径
    if (Test-Path "../CMakeLists.txt") {
        # 在 docker 目录中，相对路径应该相对于项目根目录解析
        Write-Info "当前在 docker 目录，解析相对路径: $HostPath"
        
        # 如果是相对路径，先转换为绝对路径
        if ([System.IO.Path]::IsPathRooted($HostPath)) {
            $absolutePath = $HostPath
        } else {
            # 相对路径相对于项目根目录解析（docker 目录的上级目录）
            $projectRoot = Split-Path $originalLocation -Parent
            $absolutePath = Join-Path $projectRoot $HostPath
        }
        
        # 切换到项目根目录进行构建
        Set-Location ".."
    } else {
        # 在项目根目录中
        Write-Info "当前在项目根目录，解析路径: $HostPath"
        if ([System.IO.Path]::IsPathRooted($HostPath)) {
            $absolutePath = $HostPath
        } else {
            $absolutePath = Join-Path $originalLocation $HostPath
        }
    }
    
    try {
        # 创建主机输出目录
        Write-Info "解析的绝对路径: $absolutePath"
        
        if (-not (Test-Path $absolutePath)) {
            Write-Info "创建输出目录: $absolutePath"
            New-Item -ItemType Directory -Path $absolutePath -Force | Out-Null
        }
        
        # 确保路径存在并获取规范化的绝对路径
        $absolutePath = (Get-Item $absolutePath).FullName
        
        Write-Info "主机挂载路径: $absolutePath"
        Write-Info "容器内路径: /output"
        
        # 使用挂载运行构建容器
        $mountArg = "${absolutePath}:/output"
        Write-Info "Docker 挂载参数: -v $mountArg"
        
        $buildResult = docker run --rm -v "${PWD}:/workspace" -v $mountArg -w /workspace j2-orbit-propagator-alpine:latest sh -c @"
echo '=== 开始挂载模式构建 ==='
echo '主机挂载路径: $absolutePath'
echo '容器内输出路径: /output'
echo '当前工作目录: '`$(pwd)`
echo '=== 复制构建产物到挂载目录 ==='
cp -v /usr/local/lib/*.so /output/ 2>/dev/null || echo '没有找到 .so 文件'
cp -v /usr/local/lib/*.a /output/ 2>/dev/null || echo '没有找到 .a 文件'
echo '=== 挂载目录内容 ==='
ls -la /output/
echo '=== 构建产物路径信息 ==='
find /output -type f | while read file; do
    echo "  挂载文件: `$file (容器内路径)"
    echo "  主机路径: $absolutePath/`$(basename `$file)"
    echo "  文件大小: `$(stat -c%s "`$file") bytes"
done
echo '=== 挂载模式构建完成 ==='
"@ 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "挂载模式构建完成"
            Write-Info "构建产物已直接输出到: $absolutePath"
            
            # 显示主机目录中的文件
            $files = Get-ChildItem $absolutePath -ErrorAction SilentlyContinue
            if ($files) {
                Write-Host "主机目录中的构建产物:" -ForegroundColor Cyan
                $files | Format-Table Name, Length, LastWriteTime -AutoSize
            }
            return $true
        } else {
            Write-Error "挂载模式构建失败"
            Write-Host $buildResult -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Error "挂载构建过程中出现异常: $($_.Exception.Message)"
        return $false
    }
}

# 验证构建结果
function Test-BuildResult {
    Write-Info "验证构建结果..."
    
    try {
        # 运行容器并检查库文件
        $verifyScript = @"
echo '=== 验证库文件 ==='
ls -la /usr/local/lib/
echo '=== 验证动态链接 ==='
ldd /usr/local/lib/*.so 2>/dev/null || echo '没有找到 .so 文件'
echo '=== 验证完成 ==='
"@
        
        $result = docker run --rm j2-orbit-propagator-alpine:latest sh -c $verifyScript 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "构建验证完成"
            Write-Host $result -ForegroundColor Gray
        } else {
            Write-Warning "验证过程中出现问题，但构建可能仍然成功"
        }
    }
    catch {
        Write-Warning "验证过程中出现异常: $($_.Exception.Message)"
    }
}

# 提取构建产物
function Export-Artifacts {
    Write-Info "提取构建产物到本地..."
    
    try {
        # 创建输出目录 - 统一输出到 build/Release 目录
        $outputDir = "..\build\Release"
        if (-not (Test-Path $outputDir)) {
            New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        }
        
        # 创建临时容器并复制文件
        $containerId = docker create j2-orbit-propagator-alpine:latest 2>$null
        
        if ($containerId) {
            # 复制库文件
            docker cp "${containerId}:/usr/local/lib/." $outputDir 2>$null
            
            # 删除临时容器
            docker rm $containerId 2>$null | Out-Null
            
            # 显示提取的文件
            $files = Get-ChildItem $outputDir -ErrorAction SilentlyContinue
            if ($files) {
                Write-Success "构建产物已提取到 $outputDir\"
                Write-Host "提取的文件:"
                $files | Format-Table Name, Length, LastWriteTime -AutoSize
            } else {
                Write-Warning "没有找到构建产物"
            }
        } else {
            Write-Warning "无法创建临时容器来提取文件"
        }
    }
    catch {
        Write-Warning "提取构建产物时出现异常: $($_.Exception.Message)"
    }
}

# 主函数
function Main {
    if ($Help) {
        Show-Usage
        return
    }
    
    Write-Info "开始 Alpine Linux 构建流程..."
    
    # 检查 Docker 环境
    if (-not (Test-Docker)) {
        return
    }
    
    # 处理基础镜像
    if ($BuildBaseOnly) {
        Write-Info "=== 仅构建基础镜像模式 ==="
        if (-not (Build-BaseImage)) {
            return
        }
        Write-Success "基础镜像构建完成!"
        Write-Info "基础镜像名称: cpp-dev-base:latest"
        Write-Info "包含依赖: cmake, git, pkgconfig, eigen-dev, linux-headers"
        return
    }
    
    # 检查或构建基础镜像
    if (-not $SkipBase) {
        if (-not (Test-BaseImage)) {
            Write-Info "基础镜像不存在，开始构建..."
            if (-not (Build-BaseImage)) {
                Write-Error "基础镜像构建失败，无法继续"
                return
            }
        } else {
            Write-Success "基础镜像已存在: cpp-dev-base:latest"
        }
    } else {
        Write-Info "跳过基础镜像检查（使用 -SkipBase 参数）"
    }
    
    # 执行构建步骤
    if (-not $NoCleanup) {
        Clear-OldArtifacts
    }
    
    if (-not (Build-Image)) {
        return
    }
    
    # 根据模式选择不同的输出方式
    if ($WithMount) {
        Write-Info "=== 使用挂载模式输出构建产物 ==="
        if (-not (Build-WithMount -HostPath $MountPath)) {
            return
        }
        
        Write-Success "挂载模式构建完成!"
        Write-Info "镜像名称: j2-orbit-propagator-alpine:latest"
        Write-Info "构建产物直接输出到: $MountPath"
        Write-Host "=== 动态库挂载路径信息 ===" -ForegroundColor Cyan
        Write-Host "容器内路径: /output" -ForegroundColor Gray
        Write-Host "主机挂载路径: $(Resolve-Path $MountPath)" -ForegroundColor Gray
        Write-Host "使用方式: 直接从主机路径使用库文件" -ForegroundColor Gray
    } else {
        Test-BuildResult
        
        if (-not $NoExtract) {
            Export-Artifacts
        }
        
        Write-Success "Alpine Linux 构建流程完成!"
        Write-Info "镜像名称: j2-orbit-propagator-alpine:latest"
        
        if (-not $NoExtract) {
            Write-Info "构建产物位置: build\Release\"
        }
        
        Write-Host "=== 动态库路径信息 ===" -ForegroundColor Cyan
        Write-Host "容器内路径: /usr/local/lib/" -ForegroundColor Gray
        Write-Host "提取到主机路径: build\Release\" -ForegroundColor Gray
        Write-Host "如需挂载模式，请使用: .\build.ps1 -WithMount" -ForegroundColor Gray
    }
}

# 运行主函数
Main