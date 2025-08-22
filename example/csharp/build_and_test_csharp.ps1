param(
    [string]$BuildType = "Release",
    [string]$NativeBuildDir = "..\..\build"
)

$ErrorActionPreference = 'Stop'

function Ensure-Native-Build {
    param([string]$Config)
    $buildDir = Join-Path $PSScriptRoot $NativeBuildDir
    if (-not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }
    Push-Location $buildDir
    try {
        cmake -DCMAKE_BUILD_TYPE=$Config -DBUILD_SHARED_LIBS=ON .. | Write-Host
        cmake --build . --config $Config --target j2_orbit_propagator_shared | Write-Host
    } finally { Pop-Location }

    # 复制生成的动态库到当前脚本目录，方便C#运行时加载
    $dllName = "j2_orbit_propagator.dll"
    $soName = "libj2_orbit_propagator.so"
    $nativeBinWin = Join-Path $buildDir "$Config"
    $nativeBinLinux = $buildDir

    if (Test-Path (Join-Path $nativeBinWin $dllName)) {
        Copy-Item (Join-Path $nativeBinWin $dllName) $PSScriptRoot -Force
    } elseif (Test-Path (Join-Path $nativeBinLinux $soName)) {
        Copy-Item (Join-Path $nativeBinLinux $soName) $PSScriptRoot -Force
    } else {
        Write-Host "Warning: native library not found in $nativeBinWin or $nativeBinLinux" -ForegroundColor Yellow
    }
}

function Build-CSharp {
    Push-Location $PSScriptRoot
    try {
        # 改为编译库项目，避免与独立的 J2Orbit.csproj 中重复类型发生潜在冲突
        dotnet build .\J2Orbit.Library\J2Orbit.Library.csproj -c $BuildType | Write-Host
        dotnet build .\MemoryLayoutTest\MemoryLayoutTest.csproj -c $BuildType | Write-Host
        # 如需构建示例应用，可取消注释下一行
        # dotnet build .\J2Orbit.TestApp\J2Orbit.TestApp.csproj -c $BuildType | Write-Host
    } finally { Pop-Location }
}

function Run-Tests {
    # 确保将原生库复制到测试输出目录
    $testProjDir = Join-Path $PSScriptRoot 'MemoryLayoutTest'
    $outDir = Join-Path $testProjDir "bin/$BuildType/net8.0"
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    $dll = Join-Path $PSScriptRoot 'j2_orbit_propagator.dll'
    $so = Join-Path $PSScriptRoot 'libj2_orbit_propagator.so'
    if (Test-Path $dll) { Copy-Item $dll $outDir -Force }
    if (Test-Path $so) { Copy-Item $so $outDir -Force }

    Push-Location $testProjDir
    try {
        dotnet run -c $BuildType | Write-Host
    } finally { Pop-Location }
}

Ensure-Native-Build -Config $BuildType
Build-CSharp
Run-Tests