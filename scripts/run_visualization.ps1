# =============================================================================
# J2轨道传播器可视化程序启动脚本
# 功能：自动编译并启动轨道可视化演示程序
# 作者：SOLO Coding
# 版本：1.0
# =============================================================================

param(
    [switch]$Clean,          # 清理build缓存（保留CMakeLists.txt）
    [double]$TimeScale = 1.0, # 时间缩放因子（默认1.0为实时）
    [switch]$Help            # 显示帮助信息
)

# 显示帮助信息
if ($Help) {
    Write-Host "J2轨道传播器可视化程序启动脚本" -ForegroundColor Green
    Write-Host "用法: .\run_visualization.ps1 [选项]" -ForegroundColor Yellow
    Write-Host "选项:"
    Write-Host "  -Clean       清理build缓存（保留CMakeLists.txt文件）"
    Write-Host "  -TimeScale   时间缩放因子（默认1.0为实时，>1.0加速，<1.0减速）"
    Write-Host "  -Help        显示此帮助信息"
    Write-Host "示例:"
    Write-Host "  .\run_visualization.ps1                    # 正常编译并运行（实时）"
    Write-Host "  .\run_visualization.ps1 -TimeScale 10.0    # 10倍速运行"
    Write-Host "  .\run_visualization.ps1 -TimeScale 0.5     # 0.5倍速运行"
    Write-Host "  .\run_visualization.ps1 -Clean             # 清理缓存后编译运行"
    exit 0
}

# 设置错误处理
$ErrorActionPreference = "Stop"

# 验证TimeScale参数
if ($TimeScale -le 0) {
    Write-Host "❌ 错误：TimeScale参数必须大于0" -ForegroundColor Red
    Write-Host "当前值：$TimeScale" -ForegroundColor Yellow
    Write-Host "请使用正数值，例如：1.0（实时）、10.0（10倍速）、0.5（0.5倍速）" -ForegroundColor Yellow
    exit 1
}

if ($TimeScale -gt 1000) {
    Write-Host "⚠️ 警告：TimeScale值过大（$TimeScale），可能导致数值不稳定" -ForegroundColor Yellow
    Write-Host "建议使用较小的值（如1.0-100.0）" -ForegroundColor Yellow
}

# 获取项目根目录
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VisualizationDir = Join-Path $ProjectRoot "visualization"
$BuildDir = Join-Path $VisualizationDir "build"
$ExampleDir = Join-Path $VisualizationDir "examples"

Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host "J2轨道传播器可视化程序启动脚本" -ForegroundColor Green
Write-Host "=============================================================================" -ForegroundColor Cyan

try {
    # 检查可视化目录是否存在
    if (-not (Test-Path $VisualizationDir)) {
        Write-Host "❌ 错误：可视化目录不存在: $VisualizationDir" -ForegroundColor Red
        exit 1
    }

    Write-Host "📁 项目根目录: $ProjectRoot" -ForegroundColor Yellow
    Write-Host "📁 可视化目录: $VisualizationDir" -ForegroundColor Yellow
    Write-Host "📁 构建目录: $BuildDir" -ForegroundColor Yellow
    Write-Host "⏱️ 时间缩放因子: $TimeScale" -ForegroundColor Yellow
    Write-Host ""

    # 步骤1：处理build目录
    Write-Host "🔧 步骤1：准备构建环境" -ForegroundColor Green
    
    if ($Clean -and (Test-Path $BuildDir)) {
        Write-Host "🧹 清理build缓存..." -ForegroundColor Yellow
        
        # 保存CMakeLists.txt文件
        $CMakeListsPath = Join-Path $BuildDir "CMakeLists.txt"
        $TempCMakeListsPath = Join-Path $env:TEMP "CMakeLists_backup.txt"
        
        if (Test-Path $CMakeListsPath) {
            Copy-Item $CMakeListsPath $TempCMakeListsPath -Force
            Write-Host "💾 已备份CMakeLists.txt" -ForegroundColor Cyan
        }
        
        # 删除build目录内容
        Get-ChildItem $BuildDir -Recurse | Remove-Item -Force -Recurse
        Write-Host "🗑️ 已清理build目录" -ForegroundColor Cyan
        
        # 恢复CMakeLists.txt文件
        if (Test-Path $TempCMakeListsPath) {
            Copy-Item $TempCMakeListsPath $CMakeListsPath -Force
            Remove-Item $TempCMakeListsPath -Force
            Write-Host "📋 已恢复CMakeLists.txt" -ForegroundColor Cyan
        }
    }
    
    # 创建build目录（如果不存在）
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
        Write-Host "📁 已创建build目录" -ForegroundColor Cyan
    }
    
    Write-Host "✅ 构建环境准备完成" -ForegroundColor Green
    Write-Host ""

    # 步骤2：CMake配置
    Write-Host "🔧 步骤2：CMake配置项目" -ForegroundColor Green
    Set-Location $BuildDir
    
    Write-Host "⚙️ 正在配置CMake..." -ForegroundColor Yellow
    $cmakeArgs = @(
        "..",
        "-DBUILD_VISUALIZATION=ON",
        "-DCMAKE_BUILD_TYPE=Release"
    )
    
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake配置失败"
    }
    
    Write-Host "✅ CMake配置完成" -ForegroundColor Green
    Write-Host ""

    # 步骤3：编译项目
    Write-Host "🔨 步骤3：编译可视化程序" -ForegroundColor Green
    
    Write-Host "⚙️ 正在编译项目..." -ForegroundColor Yellow
    & cmake --build . --config Release --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "项目编译失败"
    }
    
    Write-Host "✅ 编译完成" -ForegroundColor Green
    Write-Host ""

    # 步骤4：检查可执行文件
    Write-Host "🔍 步骤4：检查可执行文件" -ForegroundColor Green
    
    $ExeFiles = @(
        "bin\Release\orbit_visualization_demo.exe",
        "examples\Release\orbit_visualization_demo.exe",
        "examples\orbit_visualization_demo.exe",
        "Release\orbit_visualization_demo.exe",
        "orbit_visualization_demo.exe"
    )
    
    $ExePath = $null
    foreach ($ExeFile in $ExeFiles) {
        $TestPath = Join-Path $BuildDir $ExeFile
        if (Test-Path $TestPath) {
            $ExePath = $TestPath
            break
        }
    }
    
    if (-not $ExePath) {
        Write-Host "❌ 错误：找不到orbit_visualization_demo.exe可执行文件" -ForegroundColor Red
        Write-Host "请检查编译是否成功完成" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "📍 找到可执行文件: $ExePath" -ForegroundColor Cyan
    Write-Host "✅ 可执行文件检查完成" -ForegroundColor Green
    Write-Host ""

    # 步骤5：启动可视化程序
    Write-Host "🚀 步骤5：启动可视化程序" -ForegroundColor Green
    Write-Host "=============================================================================" -ForegroundColor Cyan
    Write-Host "正在启动J2轨道传播器可视化演示程序..." -ForegroundColor Yellow
    Write-Host "程序功能：" -ForegroundColor Cyan
    Write-Host "  • ISS国际空间站轨道可视化" -ForegroundColor White
    Write-Host "  • GEO地球同步轨道可视化" -ForegroundColor White
    Write-Host "  • Polar极地轨道可视化" -ForegroundColor White
    Write-Host "  • 实时轨道传播和J2摄动效应" -ForegroundColor White
    Write-Host "  • 轨道元素实时显示" -ForegroundColor White
    Write-Host "=============================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    # 启动程序（传递TimeScale参数）
    & $ExePath $TimeScale
    
    Write-Host ""
    Write-Host "🎉 可视化程序已退出" -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "❌ 错误：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "请检查以下项目：" -ForegroundColor Yellow
    Write-Host "  1. 确保已安装CMake和C++编译器" -ForegroundColor White
    Write-Host "  2. 确保所有依赖库已正确安装" -ForegroundColor White
    Write-Host "  3. 检查项目目录结构是否完整" -ForegroundColor White
    Write-Host "  4. 查看上方的详细错误信息" -ForegroundColor White
    exit 1
} finally {
    # 返回原始目录
    Set-Location $ProjectRoot
}

Write-Host ""
Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host "脚本执行完成！" -ForegroundColor Green
Write-Host "=============================================================================" -ForegroundColor Cyan