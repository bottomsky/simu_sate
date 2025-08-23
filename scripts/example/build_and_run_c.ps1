#!/usr/bin/env pwsh
# PowerShell脚本：编译并运行C语言示例

param(
    [string]$Compiler = "gcc",
    [switch]$Help
)

if ($Help) {
    Write-Host "用法: ./build_and_run_c.ps1 [-Compiler <compiler>] [-Help]"
    Write-Host "参数:"
    Write-Host "  -Compiler      指定编译器 (默认: gcc)"
    Write-Host "  -Help          显示此帮助信息"
    exit 0
}

Write-Host "C语言 J2轨道传播器示例编译运行脚本" -ForegroundColor Green
Write-Host "=" * 50

# 检查编译器是否可用
try {
    $compilerVersion = & $Compiler --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "编译器不可用"
    }
    Write-Host "发现编译器: $Compiler" -ForegroundColor Yellow
    Write-Host ($compilerVersion | Select-Object -First 1)
} catch {
    Write-Error "编译器 $Compiler 未安装或不在PATH中"
    Write-Host "请安装GCC或其他C编译器"
    exit 1
}

# 检查动态库文件是否存在
$dllPath = "j2_orbit_propagator.dll"
if (-not (Test-Path $dllPath)) {
    Write-Error "未找到动态库文件: $dllPath"
    Write-Host "请先运行构建脚本生成动态库"
    exit 1
}

Write-Host "找到动态库文件: $dllPath" -ForegroundColor Green

# 检查头文件是否存在
$headerPath = "../../include/j2_orbit_propagator_c.h"
if (-not (Test-Path $headerPath)) {
    Write-Error "未找到头文件: $headerPath"
    exit 1
}

Write-Host "找到头文件: $headerPath" -ForegroundColor Green

# 检查 C 示例文件是否存在
$cSourcePath = "../../example/c_example.c"
if (-not (Test-Path $cSourcePath)) {
    Write-Error "未找到 C 示例文件: $cSourcePath"
    exit 1
}

Write-Host "找到 C 示例文件: $cSourcePath" -ForegroundColor Green

# 编译C示例
Write-Host "编译C示例..." -ForegroundColor Yellow
try {
    # Windows平台编译命令
    $compileCmd = "$Compiler -I ../../include -o c_example.exe $cSourcePath -L../../bin -lj2_orbit_propagator -lm"
    Write-Host "执行编译命令: $compileCmd"
    
    Invoke-Expression $compileCmd
    if ($LASTEXITCODE -ne 0) {
        throw "编译失败"
    }
    
    Write-Host "编译成功！" -ForegroundColor Green
    
    # 检查可执行文件是否生成
    if (-not (Test-Path "c_example.exe")) {
        throw "可执行文件未生成"
    }
    
    # 运行示例
    Write-Host "运行C示例..." -ForegroundColor Yellow
    Write-Host "-" * 50
    
    ./c_example.exe
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "-" * 50
        Write-Host "C示例运行成功！" -ForegroundColor Green
    } else {
        Write-Error "C示例运行失败"
    }
    
} catch {
    Write-Error "编译或运行过程中发生错误: $_"
    Write-Host "\n可能的解决方案:"
    Write-Host "1. 确保已安装GCC编译器"
    Write-Host "2. 确保动态库文件存在于当前目录"
    Write-Host "3. 检查头文件路径是否正确"
    exit 1
} finally {
    # 清理生成的可执行文件
    if (Test-Path "c_example.exe") {
        Write-Host "清理临时文件..." -ForegroundColor Yellow
        Remove-Item "c_example.exe" -Force
    }
}

Write-Host "脚本执行完成！" -ForegroundColor Green