# J2轨道传播器动态库构建脚本
# 该脚本用于在Windows平台上构建J2轨道传播器的动态库

<#
.SYNOPSIS
 构建 J2 轨道传播器的动态库与测试。
.DESCRIPTION
 使用 CMake 在 Windows 平台生成并构建项目，支持可选清理与安装步骤。
 支持为 CMake 指定构建类型与生成器，并可将构建产物安装到指定前缀目录。
 增强的跨平台特性支持，包括 CUDA、多种生成器和并行构建选项。
.PARAMETER BuildType
 构建类型（别名: -t, -config）。可选: Debug, Release, RelWithDebInfo, MinSizeRel。默认: Release。
.PARAMETER Generator
 CMake 生成器（别名: -g）。默认: "Visual Studio 17 2022"。
 其他选项: "Ninja", "MinGW Makefiles", "Visual Studio 16 2019"等。
.PARAMETER Clean
 在配置/构建前清理构建目录（别名: -c）。
.PARAMETER Install
 构建后执行安装（别名: -i）。
.PARAMETER InstallPrefix
 安装前缀目录（别名: -p, -prefix）。默认: $PWD\install。
.PARAMETER BuildDir
 构建目录（别名: -b）。默认: build。可指定为相对或绝对路径。
.PARAMETER EnableCuda
 强制启用 CUDA 支持。
.PARAMETER DisableTests
 禁用测试构建。
.PARAMETER DisableExamples
 禁用示例构建。
.PARAMETER Jobs
 并行构建作业数（别名: -j）。默认: 自动检测处理器核心数。
.EXAMPLE
 ./build_dynamic_library.ps1 -t Release -g "Visual Studio 17 2022" -c -i -p .\install
.EXAMPLE
 ./build_dynamic_library.ps1 -config Debug -i -b build -EnableCuda -j 8
.EXAMPLE
 ./build_dynamic_library.ps1 -g Ninja -t Release -c
#>

param(
    [Alias('t','config')][ValidateSet('Debug','Release','RelWithDebInfo','MinSizeRel')][string]$BuildType = "Release",
    [Alias('g')][string]$Generator = "Visual Studio 17 2022",
    [Alias('c')][switch]$Clean,
    [Alias('i')][switch]$Install,
    [Alias('p','prefix')][string]$InstallPrefix = "$PWD\install",
    [Alias('b')][string]$BuildDir = "build",
    [switch]$EnableCuda,
    [switch]$DisableTests,
    [switch]$DisableExamples,
    [Alias('j')][int]$Jobs = 0
)

$ErrorActionPreference = "Stop"

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=== J2 轨道传播器 Windows 构建脚本 ===" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

# 自动检测并行作业数
if ($Jobs -eq 0) {
    $Jobs = [Environment]::ProcessorCount
    Write-Host "自动检测到 $Jobs 个处理器核心" -ForegroundColor Yellow
}

# 检查CMake是否可用
try {
    $cmakeVersion = cmake --version
    Write-Host "发现CMake: $($cmakeVersion[0])" -ForegroundColor Yellow
} catch {
    Write-Error "未找到CMake，请确保CMake已安装并在PATH中"
    exit 1
}

# 创建构建目录
# 使用参数化的 BuildDir 以便与其他脚本统一
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "清理构建目录: $BuildDir" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

if (-not (Test-Path $BuildDir)) {
    Write-Host "创建构建目录: $BuildDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

Set-Location $BuildDir

# 配置CMake
Write-Host "配置CMake项目..." -ForegroundColor Yellow
Write-Host "构建类型: $BuildType" -ForegroundColor Cyan
Write-Host "生成器: $Generator" -ForegroundColor Cyan
Write-Host "构建目录: $BuildDir" -ForegroundColor Cyan
Write-Host "并行作业数: $Jobs" -ForegroundColor Cyan
Write-Host "CUDA 支持: $(if ($EnableCuda) { '启用' } else { '自动检测' })" -ForegroundColor Cyan
Write-Host "构建测试: $(if ($DisableTests) { '禁用' } else { '启用' })" -ForegroundColor Cyan
Write-Host "构建示例: $(if ($DisableExamples) { '禁用' } else { '启用' })" -ForegroundColor Cyan

$cmakeArgs = @(
    "..",
    "-G", $Generator,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DBUILD_EXAMPLES=$(if ($DisableExamples) { 'OFF' } else { 'ON' })",
    "-DBUILD_TESTS=$(if ($DisableTests) { 'OFF' } else { 'ON' })"
)

if ($EnableCuda) {
    $cmakeArgs += "-DENABLE_CUDA=ON"
}

if ($Install) {
    $cmakeArgs += "-DCMAKE_INSTALL_PREFIX=$InstallPrefix"
}

try {
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake配置失败"
    }
} catch {
    Write-Error "CMake配置失败: $_"
    exit 1
}

# 构建项目
Write-Host "构建项目..." -ForegroundColor Yellow
try {
    & cmake --build . --config $BuildType --parallel $Jobs
    if ($LASTEXITCODE -ne 0) {
        throw "构建失败"
    }
} catch {
    Write-Error "构建失败: $_"
    exit 1
}

# 安装（如果指定）
if ($Install) {
    Write-Host "安装到: $InstallPrefix" -ForegroundColor Yellow
    try {
        & cmake --install . --config $BuildType
        if ($LASTEXITCODE -ne 0) {
            throw "安装失败"
        }
    } catch {
        Write-Error "安装失败: $_"
        exit 1
    }
}

# 显示构建结果
Write-Host "构建完成！" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

# 查找生成的文件
$dllFiles = Get-ChildItem -Recurse -Filter "*.dll" | Where-Object { $_.Name -like "*j2_orbit_propagator*" }
$libFiles = Get-ChildItem -Recurse -Filter "*.lib" | Where-Object { $_.Name -like "*j2_orbit_propagator*" }
$exeFiles = Get-ChildItem -Recurse -Filter "*.exe" | Where-Object { $_.Name -like "*j2*" -or $_.Name -like "*test*" }

if ($dllFiles) {
    Write-Host "生成的动态库文件:" -ForegroundColor Cyan
    foreach ($file in $dllFiles) {
        Write-Host "  $($file.FullName)" -ForegroundColor White
    }
}

if ($libFiles) {
    Write-Host "生成的静态库文件:" -ForegroundColor Cyan
    foreach ($file in $libFiles) {
        Write-Host "  $($file.FullName)" -ForegroundColor White
    }
}

if ($exeFiles) {
    Write-Host "生成的可执行文件:" -ForegroundColor Cyan
    foreach ($file in $exeFiles) {
        Write-Host "  $($file.FullName)" -ForegroundColor White
    }
}

# 复制动态库到bin目录
$BinDir = "..\bin"
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir | Out-Null
}

if ($dllFiles) {
    Write-Host "复制动态库到bin目录..." -ForegroundColor Yellow
    foreach ($dll in $dllFiles) {
        $destPath = Join-Path $BinDir $dll.Name
        Copy-Item $dll.FullName $destPath -Force
        Write-Host "  复制: $($dll.Name) -> $destPath" -ForegroundColor White
    }
}

if ($libFiles) {
    Write-Host "复制静态库到bin目录..." -ForegroundColor Yellow
    foreach ($lib in $libFiles) {
        $destPath = Join-Path $BinDir $lib.Name
        Copy-Item $lib.FullName $destPath -Force
        Write-Host "  复制: $($lib.Name) -> $destPath" -ForegroundColor White
    }
}

if ($exeFiles) {
    Write-Host "复制可执行文件到bin目录..." -ForegroundColor Yellow
    foreach ($exe in $exeFiles) {
        $destPath = Join-Path $BinDir $exe.Name
        Copy-Item $exe.FullName $destPath -Force
        Write-Host "  复制: $($exe.Name) -> $destPath" -ForegroundColor White
    }
}

# 复制动态库到示例目录（保持向后兼容）
$ExampleDir = "..\example"
if ($dllFiles -and (Test-Path $ExampleDir)) {
    Write-Host "复制动态库到示例目录..." -ForegroundColor Yellow
    foreach ($dll in $dllFiles) {
        $destPath = Join-Path $ExampleDir $dll.Name
        Copy-Item $dll.FullName $destPath -Force
        Write-Host "  复制: $($dll.Name) -> $destPath" -ForegroundColor White
    }
}

Write-Host "\n使用说明:" -ForegroundColor Green
Write-Host "1. Python绑定示例: python example\python_binding_example.py" -ForegroundColor White
Write-Host "2. C#绑定示例: 编译并运行 example\CSharpBindingExample.cs" -ForegroundColor White
Write-Host "3. 动态库文件可用于其他语言的FFI调用" -ForegroundColor White

Set-Location ..
Write-Host "构建脚本执行完成！" -ForegroundColor Green