<#
.SYNOPSIS
    构建和运行 J2 轨道传播器的 Vulkan 可视化演示程序

.DESCRIPTION
    此脚本用于：
    1. 检查 Vulkan SDK 是否已安装
    2. 配置 CMake 并启用可视化模块
    3. 编译项目
    4. 运行可视化演示程序

.PARAMETER Clean
    清理构建缓存（保留 CMakeLists.txt）

.PARAMETER BuildType
    构建类型：Debug 或 Release（默认：Debug）

.PARAMETER SkipBuild
    跳过编译，直接运行演示程序

.EXAMPLE
    .\build_visualization.ps1
    .\build_visualization.ps1 -Clean -BuildType Release
    .\build_visualization.ps1 -SkipBuild
#>

param(
    [switch]$Clean,
    [ValidateSet("Debug", "Release")]
    [string]$BuildType = "Debug",
    [switch]$SkipBuild,
    [string]$CMAKE_ARGS = "",
    [switch]$Help
)

# 显示帮助信息
if ($Help) {
    Write-Host "J2 轨道传播器 Vulkan 可视化构建脚本" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "用法: .\build_visualization.ps1 [参数]" -ForegroundColor White
    Write-Host ""
    Write-Host "参数:" -ForegroundColor Yellow
    Write-Host "  -Clean          清理构建缓存（保留 CMakeLists.txt）" -ForegroundColor White
    Write-Host "  -BuildType      构建类型：Debug 或 Release（默认：Debug）" -ForegroundColor White
    Write-Host "  -SkipBuild      跳过编译，直接运行演示程序" -ForegroundColor White
    Write-Host "  -Help           显示此帮助信息" -ForegroundColor White
    Write-Host ""
    Write-Host "示例:" -ForegroundColor Yellow
    Write-Host "  .\build_visualization.ps1" -ForegroundColor Green
    Write-Host "  .\build_visualization.ps1 -Clean -BuildType Release" -ForegroundColor Green
    Write-Host "  .\build_visualization.ps1 -SkipBuild" -ForegroundColor Green
    exit 0
}

# 颜色输出函数
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# 错误处理函数
function Write-Error-Exit {
    param([string]$Message)
    Write-ColorOutput "错误: $Message" "Red"
    exit 1
}

# 成功信息函数
function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✓ $Message" "Green"
}

# 信息输出函数
function Write-Info {
    param([string]$Message)
    Write-ColorOutput "ℹ $Message" "Cyan"
}

# 警告输出函数
function Write-Warning-Custom {
    param([string]$Message)
    Write-ColorOutput "⚠ $Message" "Yellow"
}

# 检查 Vulkan SDK
function Test-VulkanSDK {
    Write-Info "检查 Vulkan SDK..."
    
    # 首先检查 vulkaninfo 命令是否可用
    try {
        $vulkanInfoOutput = vulkaninfo --summary 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "找到 Vulkan 运行时"
            
            # 检查是否有完整的 SDK（开发组件）
            $vulkanPath = $env:VULKAN_SDK
            if ($vulkanPath -and (Test-Path (Join-Path $vulkanPath "Include\vulkan\vulkan.h"))) {
                Write-Success "找到完整的 Vulkan SDK: $vulkanPath"
                return $vulkanPath
            } else {
                Write-Warning-Custom "检测到 Vulkan 运行时，但未找到开发组件（头文件和库）"
                Write-Error-Exit "需要完整的 Vulkan SDK 进行开发。请从 https://vulkan.lunarg.com/ 下载并安装 Vulkan SDK，确保选择包含开发组件的完整安装。"
            }
        }
    } catch {
        # vulkaninfo 命令不可用，继续查找传统安装
    }
    
    # 检查环境变量
    $vulkanPath = $env:VULKAN_SDK
    if (-not $vulkanPath) {
        Write-Warning-Custom "未找到 VULKAN_SDK 环境变量"
        
        # 尝试在常见位置查找 Vulkan
        $commonPaths = @(
            "C:\VulkanSDK\*\Bin\vulkaninfo.exe",
            "${env:ProgramFiles}\VulkanSDK\*\Bin\vulkaninfo.exe",
            "${env:ProgramFiles(x86)}\VulkanSDK\*\Bin\vulkaninfo.exe"
        )
        
        $vulkanInfo = $null
        foreach ($path in $commonPaths) {
            $vulkanInfo = Get-ChildItem $path -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($vulkanInfo) {
                $vulkanPath = Split-Path (Split-Path $vulkanInfo.FullName)
                Write-Info "在 $vulkanPath 找到 Vulkan SDK"
                break
            }
        }
        
        if (-not $vulkanInfo) {
            Write-Error-Exit "未找到完整的 Vulkan SDK。检测到系统中有 Vulkan 运行时，但缺少开发组件（头文件和库）。请从 https://vulkan.lunarg.com/ 下载并安装完整的 Vulkan SDK。"
        }
    } else {
        Write-Success "找到 Vulkan SDK: $vulkanPath"
    }
    
    # 验证 vulkaninfo 工具
    if ($vulkanPath -ne "System") {
        $vulkanInfoPath = Join-Path $vulkanPath "Bin\vulkaninfo.exe"
        if (-not (Test-Path $vulkanInfoPath)) {
            Write-Error-Exit "Vulkan SDK 安装不完整，未找到 vulkaninfo.exe"
        }
    }
    
    Write-Success "Vulkan SDK 验证通过"
    return $vulkanPath
}

# 检查必要工具
function Test-RequiredTools {
    Write-Info "检查必要工具..."
    
    # 检查 CMake
    try {
        $cmakeVersion = cmake --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "找到 CMake"
        } else {
            Write-Error-Exit "未找到 CMake。请安装 CMake 3.15 或更高版本"
        }
    } catch {
        Write-Error-Exit "未找到 CMake。请安装 CMake 3.15 或更高版本"
    }
    
    # 检查 Visual Studio 构建工具
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $vsInstances = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json | ConvertFrom-Json
        if ($vsInstances) {
            Write-Success "找到 Visual Studio 构建工具"
        } else {
            Write-Warning-Custom "未找到 Visual Studio 构建工具，可能影响编译"
        }
    } else {
        Write-Warning-Custom "未找到 vswhere.exe，无法验证 Visual Studio 安装"
    }
    
    # 检查 vcpkg（可选）
    $vcpkgPath = Get-Command vcpkg -ErrorAction SilentlyContinue
    if ($vcpkgPath) {
        Write-Success "找到 vcpkg 包管理器"
        Write-Info "如果编译时缺少 GLFW 或 GLM，可以使用 vcpkg 安装："
        Write-Info "  vcpkg install glfw3:x64-windows glm:x64-windows"
    } else {
        Write-Warning-Custom "未找到 vcpkg。如果编译失败，请考虑安装 vcpkg 来管理依赖项"
        Write-Info "vcpkg 安装指南："
        Write-Info "  1. git clone https://github.com/Microsoft/vcpkg.git"
        Write-Info "  2. cd vcpkg && .\\bootstrap-vcpkg.bat"
        Write-Info "  3. .\\vcpkg integrate install"
        Write-Info "  4. .\\vcpkg install glfw3:x64-windows glm:x64-windows"
        Write-Info "或者手动下载并编译 GLFW 和 GLM 库"
    }
}

# 清理构建目录
function Clear-BuildCache {
    param([string]$BuildDir)
    
    Write-Info "清理构建缓存..."
    
    if (Test-Path $BuildDir) {
        # 保存 CMakeLists.txt
        $cmakeListsPath = Join-Path $BuildDir "CMakeLists.txt"
        $tempCMakeLists = $null
        if (Test-Path $cmakeListsPath) {
            $tempCMakeLists = Get-Content $cmakeListsPath
        }
        
        # 删除构建目录内容（除了 CMakeLists.txt）
        Get-ChildItem $BuildDir | Where-Object { $_.Name -ne "CMakeLists.txt" } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        
        # 恢复 CMakeLists.txt
        if ($tempCMakeLists) {
            $tempCMakeLists | Set-Content $cmakeListsPath
        }
        
        Write-Success "构建缓存已清理"
    } else {
        Write-Info "构建目录不存在，无需清理"
    }
}

# 配置 CMake
function Invoke-CMakeConfigure {
    param(
        [string]$SourceDir,
        [string]$BuildDir,
        [string]$BuildType,
        [string]$ExtraArgs = ""
    )
    
    Write-Info "配置 CMake..."
    
    # 确保构建目录存在
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
    }
    
    # CMake 配置参数
    $cmakeArgs = @(
        "-G", "Visual Studio 17 2022",
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DBUILD_EXAMPLES=ON",
        "-DBUILD_TESTS=ON",
        "-DBUILD_VISUALIZATION=ON",
        "-DENABLE_CUDA=OFF",
        $SourceDir
    )
    
    # 添加额外的 CMake 参数
    if ($ExtraArgs) {
        $extraArgsList = $ExtraArgs -split ' ' | Where-Object { $_ -ne '' }
        $cmakeArgs = $cmakeArgs + $extraArgsList
        Write-Host "额外的 CMake 参数: $ExtraArgs" -ForegroundColor Yellow
        Write-Host "解析后的参数: $($extraArgsList -join ', ')" -ForegroundColor Yellow
    }
    
    Write-Host "完整的 CMake 参数: $($cmakeArgs -join ' ')" -ForegroundColor Cyan
    
    Push-Location $BuildDir
    try {
        & cmake @cmakeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Exit "CMake 配置失败"
        }
        Write-Success "CMake 配置完成"
    } finally {
        Pop-Location
    }
}

# 编译项目
function Invoke-ProjectBuild {
    param(
        [string]$BuildDir,
        [string]$BuildType
    )
    
    Write-Info "编译项目..."
    
    Push-Location $BuildDir
    try {
        & cmake --build . --config $BuildType --target orbit_visualization_demo
        if ($LASTEXITCODE -ne 0) {
            Write-Error-Exit "项目编译失败"
        }
        Write-Success "项目编译完成"
    } finally {
        Pop-Location
    }
}

# 运行可视化演示
function Start-VisualizationDemo {
    param(
        [string]$BuildDir,
        [string]$BuildType
    )
    
    Write-Info "启动可视化演示程序..."
    
    # 查找可执行文件
    $exePaths = @(
        Join-Path $BuildDir "$BuildType\orbit_visualization_demo.exe",
        Join-Path $BuildDir "visualization\examples\$BuildType\orbit_visualization_demo.exe",
        Join-Path $BuildDir "bin\$BuildType\orbit_visualization_demo.exe"
    )
    
    $exePath = $null
    foreach ($path in $exePaths) {
        if (Test-Path $path) {
            $exePath = $path
            break
        }
    }
    
    if (-not $exePath) {
        Write-Error-Exit "未找到可视化演示程序可执行文件"
    }
    
    Write-Success "找到可执行文件: $exePath"
    Write-Info "正在启动可视化演示程序..."
    Write-ColorOutput "按 ESC 键退出演示程序" "Yellow"
    
    try {
        & $exePath
        if ($LASTEXITCODE -ne 0) {
            Write-Warning-Custom "演示程序退出，退出码: $LASTEXITCODE"
        } else {
            Write-Success "演示程序正常退出"
        }
    } catch {
        Write-Error-Exit "启动演示程序时发生错误: $($_.Exception.Message)"
    }
}

# 主函数
function Main {
    Write-ColorOutput "=== J2 轨道传播器 Vulkan 可视化构建脚本 ===" "Magenta"
    Write-ColorOutput ""
    
    # 获取项目根目录
    $scriptPath = $MyInvocation.MyCommand.Path
    if (-not $scriptPath) {
        $scriptPath = $PSCommandPath
    }
    if (-not $scriptPath) {
        $scriptPath = (Get-Location).Path + "\build_visualization.ps1"
    }
    
    $scriptDir = Split-Path -Parent $scriptPath
    $projectRoot = Split-Path -Parent $scriptDir
    $buildDir = Join-Path $projectRoot "build"
    
    Write-Info "项目根目录: $projectRoot"
    Write-Info "构建目录: $buildDir"
    Write-Info "构建类型: $BuildType"
    Write-ColorOutput ""
    
    try {
        # 检查 Vulkan SDK
        Test-VulkanSDK | Out-Null
        Write-ColorOutput ""
        
        # 检查必要工具
        Test-RequiredTools
        Write-ColorOutput ""
        
        if (-not $SkipBuild) {
            # 清理构建缓存（如果需要）
            if ($Clean) {
                Clear-BuildCache -BuildDir $buildDir
                Write-ColorOutput ""
            }
            
            # 配置 CMake
            Invoke-CMakeConfigure -SourceDir $projectRoot -BuildDir $buildDir -BuildType $BuildType -ExtraArgs $CMAKE_ARGS
            Write-ColorOutput ""
            
            # 编译项目
            Invoke-ProjectBuild -BuildDir $buildDir -BuildType $BuildType
            Write-ColorOutput ""
        } else {
            Write-Info "跳过编译步骤"
            Write-ColorOutput ""
        }
        
        # 运行可视化演示
        Start-VisualizationDemo -BuildDir $buildDir -BuildType $BuildType
        
    } catch {
        Write-Error-Exit "脚本执行过程中发生未预期的错误: $($_.Exception.Message)"
    }
    
    Write-ColorOutput ""
    Write-Success "脚本执行完成"
}

# 执行主函数
Main