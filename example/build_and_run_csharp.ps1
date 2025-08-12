#!/usr/bin/env pwsh
# PowerShell脚本：编译并运行C#绑定示例

param(
    [string]$DotNetVersion = "net6.0",
    [switch]$Help
)

if ($Help) {
    Write-Host "用法: ./build_and_run_csharp.ps1 [-DotNetVersion <version>] [-Help]"
    Write-Host "参数:"
    Write-Host "  -DotNetVersion  指定.NET版本 (默认: net8.0)"
    Write-Host "  -Help          显示此帮助信息"
    exit 0
}

Write-Host "C# J2轨道传播器绑定示例编译运行脚本" -ForegroundColor Green
Write-Host "=" * 50

# 检查.NET是否安装
try {
    $dotnetVersion = dotnet --version
    Write-Host "发现.NET版本: $dotnetVersion" -ForegroundColor Yellow
    
    # 根据安装的.NET版本自动选择合适的目标框架
    $majorVersion = [int]($dotnetVersion.Split('.')[0])
    if ($majorVersion -ge 8) {
        $DotNetVersion = "net8.0"
    } elseif ($majorVersion -ge 6) {
        $DotNetVersion = "net6.0"
    } else {
        $DotNetVersion = "netcoreapp3.1"
    }
    Write-Host "使用目标框架: $DotNetVersion" -ForegroundColor Yellow
} catch {
    Write-Error ".NET未安装或不在PATH中，请先安装.NET SDK"
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

# 编译C#示例
Write-Host "编译C#示例..." -ForegroundColor Yellow
try {
    # 创建临时项目文件
    $csprojContent = @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>$DotNetVersion</TargetFramework>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
"@
    
    $csprojContent | Out-File -FilePath "CSharpBindingExample.csproj" -Encoding UTF8
    
    # 编译项目
    dotnet build --configuration Release
    if ($LASTEXITCODE -ne 0) {
        throw "编译失败"
    }
    
    Write-Host "编译成功！" -ForegroundColor Green
    
    # 运行示例
    Write-Host "运行C#示例..." -ForegroundColor Yellow
    Write-Host "-" * 50
    
    dotnet run --configuration Release
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "-" * 50
        Write-Host "C#示例运行成功！" -ForegroundColor Green
    } else {
        Write-Error "C#示例运行失败"
    }
    
} catch {
    Write-Error "编译或运行过程中发生错误: $_"
    exit 1
} finally {
    # 清理临时文件
    if (Test-Path "CSharpBindingExample.csproj") {
        Remove-Item "CSharpBindingExample.csproj" -Force
    }
    if (Test-Path "bin") {
        Remove-Item "bin" -Recurse -Force
    }
    if (Test-Path "obj") {
        Remove-Item "obj" -Recurse -Force
    }
}

Write-Host "脚本执行完成！" -ForegroundColor Green