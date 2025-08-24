#!/usr/bin/env pwsh
# PowerShell脚本：编译并运行C#绑定示例

param(
    [string]$DotNetVersion = "net6.0",
    [switch]$DllOnly,
    [switch]$Help
)

if ($Help) {
    Write-Host "用法: ./build_and_run_csharp.ps1 [-DotNetVersion <version>] [-DllOnly] [-Help]"
    Write-Host "参数:"
    Write-Host "  -DotNetVersion  指定.NET版本 (默认: net8.0)"
    Write-Host "  -DllOnly       仅构建并复制DLL，不运行C#示例"
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

# 设置路径
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$buildReleaseDir = Join-Path $rootDir "build\Release"
$sourceDllPath = Join-Path $buildReleaseDir "j2_orbit_propagator.dll"
$buildLibraryScript = Join-Path $rootDir "scripts\build_dynamic_library.ps1"

# 检查是否需要构建C++库
if (-not (Test-Path $buildReleaseDir) -or -not (Test-Path $sourceDllPath)) {
    Write-Host "Release构建目录不存在或DLL文件缺失，开始构建C++库..." -ForegroundColor Yellow
    
    if (-not (Test-Path $buildLibraryScript)) {
        Write-Error "构建脚本未找到: $buildLibraryScript"
        exit 1
    }
    
    # 切换到根目录执行构建脚本
    Push-Location $rootDir
    try {
        Write-Host "执行构建脚本: $buildLibraryScript" -ForegroundColor Yellow
        & $buildLibraryScript -BuildType Release
        if ($LASTEXITCODE -ne 0) {
            throw "C++库构建失败"
        }
        Write-Host "C++库构建成功！" -ForegroundColor Green
    } catch {
        Write-Error "构建C++库时发生错误: $_"
        exit 1
    } finally {
        Pop-Location
    }
}

# 确认源DLL文件存在
if (-not (Test-Path $sourceDllPath)) {
    Write-Error "构建完成后仍未找到DLL文件: $sourceDllPath"
    exit 1
}

# 检查并复制DLL到当前目录（C#调试根目录）
$localDllPath = "j2_orbit_propagator.dll"
if (-not (Test-Path $localDllPath) -or (Get-Item $sourceDllPath).LastWriteTime -gt (Get-Item $localDllPath).LastWriteTime) {
    Write-Host "复制DLL文件到C#调试目录..." -ForegroundColor Yellow
    Copy-Item $sourceDllPath $localDllPath -Force
    Write-Host "DLL文件复制完成: $localDllPath" -ForegroundColor Green
} else {
    Write-Host "DLL文件已是最新版本: $localDllPath" -ForegroundColor Green
}

# 复制DLL到 C# 调试根目录 (example\csharp\bin\Debug)
$csharpDebugDir = Join-Path $scriptDir "csharp\bin\Debug"
try {
    if (-not (Test-Path $csharpDebugDir)) {
        Write-Host "创建目录: $csharpDebugDir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $csharpDebugDir | Out-Null
    }
    $targetDebugDll = Join-Path $csharpDebugDir "j2_orbit_propagator.dll"
    Copy-Item $sourceDllPath $targetDebugDll -Force
    Write-Host "DLL已复制到C#调试目录: $targetDebugDll" -ForegroundColor Green
} catch {
    Write-Warning "复制DLL到C#调试目录失败: $_"
}

# 如果只需要DLL则提前退出
if ($DllOnly) {
    Write-Host "已按要求仅构建并复制DLL，跳过C#示例编译与运行。" -ForegroundColor Yellow
    exit 0
}
 
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
    <!-- 避免与自动生成的 AssemblyInfo 冲突 -->
    <GenerateAssemblyInfo>false</GenerateAssemblyInfo>
    <!-- 禁用默认递归包含，避免包含子目录下的示例代码（如 csharp/MemoryLayoutTest） -->
    <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="CSharpBindingExample.cs" />
  </ItemGroup>
</Project>
"@
    
    $csprojContent | Out-File -FilePath "CSharpBindingExample.csproj" -Encoding UTF8
    
    # 编译项目
    dotnet build --configuration Release
    if ($LASTEXITCODE -ne 0) {
        throw "编译失败"
    }
    
    Write-Host "编译成功！" -ForegroundColor Green

    # 将原生DLL复制到输出目录，确保运行时能加载
    $outputDir = Join-Path $PWD "build/Release/$DotNetVersion"
    if (-not (Test-Path $outputDir)) {
        Write-Host "创建输出目录: $outputDir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    try {
        Copy-Item $sourceDllPath (Join-Path $outputDir "j2_orbit_propagator.dll") -Force
        Write-Host "已将DLL复制到输出目录: $outputDir" -ForegroundColor Green
    } catch {
        Write-Warning "复制DLL到输出目录失败: $_"
    }
    
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