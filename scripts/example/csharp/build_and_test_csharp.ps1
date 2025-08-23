param(
    [string]$BuildType = "Release",
    [string]$NativeBuildDir = "..\..\build",
    [switch]$CleanNative,
    [switch]$NativeReconfigure,
    [string]$NativeConfig = "",
    [ValidateSet('MemoryLayoutTest','TestApp','TestProject','None')][string]$Run = 'MemoryLayoutTest'
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

function Get-AppProjectInfo {
    <#
    .SYNOPSIS
      根据传入的目标应用名称解析 C# 项目路径与输出目录。
    .DESCRIPTION
      支持 MemoryLayoutTest / TestApp / TestProject 三种控制台项目，统一假设目标框架为 net8.0。
    .PARAMETER Name
      目标应用名称。可选值：MemoryLayoutTest、TestApp、TestProject。
    .OUTPUTS
      PSCustomObject，包含：
        - ProjectDir: 项目目录的绝对路径
        - Csproj: .csproj 文件的绝对路径
        - OutDir: 输出目录（bin/<Config>/net8.0）的绝对路径
    .EXCEPTIONS
      当传入的名称不受支持或项目文件不存在时抛出异常。
    #>
    param(
        [Parameter(Mandatory)][ValidateSet('MemoryLayoutTest','TestApp','TestProject')]
        [string]$Name,
        [string]$Config
    )

    $base = $PSScriptRoot
    $tfm = 'net8.0'

    switch ($Name) {
        'MemoryLayoutTest' {
            $projDir = Join-Path $base 'MemoryLayoutTest'
            $csproj = Join-Path $projDir 'MemoryLayoutTest.csproj'
        }
        'TestApp' {
            $projDir = Join-Path $base 'J2Orbit.TestApp'
            $csproj = Join-Path $projDir 'J2Orbit.TestApp.csproj'
        }
        'TestProject' {
            $projDir = Join-Path $base 'TestProject'
            $csproj = Join-Path $projDir 'TestProject.csproj'
        }
    }

    if (-not (Test-Path -LiteralPath $csproj)) {
        throw "C# project not found: $csproj"
    }

    if ([string]::IsNullOrWhiteSpace($Config)) { $Config = $BuildType }
    $outDir = Join-Path $projDir (Join-Path ('bin') (Join-Path $Config $tfm))

    [PSCustomObject]@{
        ProjectDir = (Resolve-Path $projDir).Path
        Csproj     = (Resolve-Path $csproj).Path
        OutDir     = $outDir
    }
}

function Build-CSharp {
    <#
    .SYNOPSIS
      构建 C# 库与选定的测试/示例项目。
    .DESCRIPTION
      始终构建 J2Orbit.Library（C# 封装库）；根据 -Run 参数选择性构建 MemoryLayoutTest / TestApp / TestProject。
    .PARAMETER TargetApp
      目标应用名称。可取值：MemoryLayoutTest、TestApp、TestProject、None。传入 None 时仅构建库。
    .OUTPUTS
      无。生成到各自项目的 bin/ 目录。
    .EXCEPTIONS
      dotnet build 出错时将抛出 PowerShell 运行时异常。
    #>
    param(
        [ValidateSet('MemoryLayoutTest','TestApp','TestProject','None')]
        [string]$TargetApp = 'MemoryLayoutTest'
    )

    Push-Location $PSScriptRoot
    try {
        dotnet build .\J2Orbit.Library\J2Orbit.Library.csproj -c $BuildType | Write-Host

        if ($TargetApp -ne 'None') {
            $info = Get-AppProjectInfo -Name $TargetApp -Config $BuildType
            dotnet build $info.Csproj -c $BuildType | Write-Host
        }
    } finally { Pop-Location }
}

function Copy-NativeLibToOutput {
    <#
    .SYNOPSIS
      将 bin/ 下的原生库复制到指定 C# 项目的输出目录。
    .DESCRIPTION
      按平台优先顺序尝试复制 j2_orbit_propagator.dll / libj2_orbit_propagator.so / libj2_orbit_propagator.dylib。
    .PARAMETER OutputDir
      目标输出目录（通常为 项目/bin/<Config>/net8.0）。
    .OUTPUTS
      无。
    .EXCEPTIONS
      如果在仓库根 bin/ 未找到任一原生库，则抛出异常。
    #>
    param(
        [Parameter(Mandatory)][string]$OutputDir
    )

    if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

    $rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    $rootBin = Join-Path $rootDir 'bin'

    $candidates = @(
        'j2_orbit_propagator.dll',
        'libj2_orbit_propagator.so',
        'libj2_orbit_propagator.dylib'
    )

    $found = $false
    foreach ($name in $candidates) {
        $p = Join-Path $rootBin $name
        if (Test-Path -LiteralPath $p) {
            Copy-Item $p $OutputDir -Force
            $found = $true
        }
    }

    if (-not $found) {
        throw "Native library not found in $rootBin"
    }
}

function Run-Tests {
    <#
    .SYNOPSIS
      运行选定的 C# 控制台程序。
    .DESCRIPTION
      从仓库根 bin/ 目录复制原生库到目标项目输出目录后执行 dotnet run。
    .PARAMETER TargetApp
      目标应用名称。可取值：MemoryLayoutTest、TestApp、TestProject。
    .OUTPUTS
      无。控制台打印测试输出。
    .EXCEPTIONS
      如果原生库未找到或运行失败将抛出 PowerShell 运行时异常。
    #>
    param(
        [Parameter(Mandatory)][ValidateSet('MemoryLayoutTest','TestApp','TestProject')]
        [string]$TargetApp
    )

    $info = Get-AppProjectInfo -Name $TargetApp -Config $BuildType
    Copy-NativeLibToOutput -OutputDir $info.OutDir

    Push-Location $info.ProjectDir
    try {
        dotnet run -c $BuildType | Write-Host
    } finally { Pop-Location }
}

# 统一执行流程
$nativeCfg = $BuildType
if (-not [string]::IsNullOrWhiteSpace($NativeConfig)) { $nativeCfg = $NativeConfig }

Ensure-Native-Build -Config $nativeCfg -Clean:$CleanNative -Reconfigure:$NativeReconfigure
Build-CSharp -TargetApp $Run
if ($Run -ne 'None') {
    Run-Tests -TargetApp $Run
} else {
    Write-Host "[Info] Skipped running any C# app (-Run None)." -ForegroundColor Yellow
}