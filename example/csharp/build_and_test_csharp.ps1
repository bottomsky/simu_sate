param(
    [string]$BuildType = "Release",
    [string]$NativeBuildDir = "..\..\build",
    [switch]$CleanNative,
    [switch]$NativeReconfigure,
    [string]$NativeConfig = ""
)

$ErrorActionPreference = 'Stop'

function Ensure-Native-Build {
    <#
    .SYNOPSIS
      使用仓库根目录的 C++ 构建脚本构建原生库（可选清理并强制重新配置）。
    .DESCRIPTION
      调用 ..\..\scripts\build.ps1 进行构建，构建产物会被收集至仓库根目录的 bin/ 目录。
    .PARAMETER Config
      原生构建配置（例如 Release/Debug）。如果未指定，将默认沿用脚本的 $BuildType。
    .PARAMETER Clean
      指定则在构建前清理 build 目录下缓存，但保留 build/CMakeLists.txt。
    .PARAMETER Reconfigure
      指定则删除 CMakeCache.txt 和 CMakeFiles/ 以强制 cmake 重新配置。
    .OUTPUTS
      无。构建产物输出到仓库根目录的 bin/。
    .EXCEPTIONS
      如果构建脚本执行失败，将抛出 PowerShell 运行时异常并终止执行。
    #>
    param(
        [string]$Config,
        [switch]$Clean,
        [switch]$Reconfigure
    )

    # 解析仓库根目录与构建脚本路径
    $rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    $buildPs1 = Join-Path $rootDir 'scripts\build.ps1'

    if (-not (Test-Path -LiteralPath $buildPs1)) {
        throw "Native build script not found: $buildPs1"
    }

    if ([string]::IsNullOrWhiteSpace($Config)) { $Config = $BuildType }

    Write-Host "[Native] Invoking root build script: $buildPs1 (Config=$Config, Clean=$Clean, Reconfigure=$Reconfigure)" -ForegroundColor Cyan

    if ($Clean) {
        & $buildPs1 -Clean -Config $Config -Reconfigure:$Reconfigure
    } else {
        & $buildPs1 -Config $Config -Reconfigure:$Reconfigure
    }
}

function Build-CSharp {
    <#
    .SYNOPSIS
      构建 C# 库与测试项目。
    .DESCRIPTION
      构建 J2Orbit.Library（C# 封装库）与 MemoryLayoutTest（示例/测试控制台）。
    .OUTPUTS
      无。生成到各自项目的 bin/ 目录。
    .EXCEPTIONS
      dotnet build 出错时将抛出 PowerShell 运行时异常。
    #>
    Push-Location $PSScriptRoot
    try {
        dotnet build .\J2Orbit.Library\J2Orbit.Library.csproj -c $BuildType | Write-Host
        dotnet build .\MemoryLayoutTest\MemoryLayoutTest.csproj -c $BuildType | Write-Host
        # 如需构建示例应用，可取消注释下一行
        # dotnet build .\J2Orbit.TestApp\J2Orbit.TestApp.csproj -c $BuildType | Write-Host
    } finally { Pop-Location }
}

function Run-Tests {
    <#
    .SYNOPSIS
      运行 MemoryLayoutTest 测试程序。
    .DESCRIPTION
      从仓库根 bin/ 目录复制原生库到测试输出目录后执行 dotnet run。
    .OUTPUTS
      无。控制台打印测试输出。
    .EXCEPTIONS
      如果原生库未找到或运行失败将抛出 PowerShell 运行时异常。
    #>
    $testProjDir = Join-Path $PSScriptRoot 'MemoryLayoutTest'
    $outDir = Join-Path $testProjDir "bin/$BuildType/net8.0"
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

    # 从仓库根 bin 目录复制原生库
    $rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    $rootBin = Join-Path $rootDir 'bin'
    $dll = Join-Path $rootBin 'j2_orbit_propagator.dll'
    $so  = Join-Path $rootBin 'libj2_orbit_propagator.so'

    if (Test-Path $dll) { Copy-Item $dll $outDir -Force }
    elseif (Test-Path $so) { Copy-Item $so $outDir -Force }
    else { throw "Native library not found in $rootBin" }

    Push-Location $testProjDir
    try {
        dotnet run -c $BuildType | Write-Host
    } finally { Pop-Location }
}

# 统一执行流程
$nativeCfg = $BuildType
if (-not [string]::IsNullOrWhiteSpace($NativeConfig)) { $nativeCfg = $NativeConfig }

Ensure-Native-Build -Config $nativeCfg -Clean:$CleanNative -Reconfigure:$NativeReconfigure
Build-CSharp
Run-Tests