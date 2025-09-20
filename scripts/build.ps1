<#
.SYNOPSIS
  Configure and build the project, with an option to clean build cache while preserving build/CMakeLists.txt.

.DESCRIPTION
  This PowerShell script configures and builds the C++ project using CMake.
  It supports cleaning the build cache (removing generated files in the build directory
  while preserving the build/CMakeLists.txt file as required by the project layout).
  The primary build artifacts reside under build/<Config> (e.g., build/Debug, build/Release).
  For convenience, the script also collects Windows runtime artifacts (DLL/EXE)
  into the repository-level bin directory.
#>

[CmdletBinding()]
param(
  [switch]$Clean,
  [switch]$CleanCache,
  [string]$Config = 'Release',
  [int]$Parallel = [Environment]::ProcessorCount,
  [switch]$Reconfigure,
  [string]$Generator = '',
  [string]$Toolchain = '',
  [switch]$EnableCuda,
  [switch]$DisableTests,
  [switch]$DisableExamples,
  [switch]$Visualization,
  [switch]$AlsoLinux
)

$ErrorActionPreference = 'Stop'

function Write-Section {
  param([string]$Message)
  Write-Host "`n==== $Message ====\n" -ForegroundColor Cyan
}

function Ensure-Directory {
  param([string]$Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

function Clean-BuildCache {
  param([string]$BuildDir)
  if (-not (Test-Path -LiteralPath $BuildDir)) {
    Write-Host "[Clean] Build directory not found, skipping clean: $BuildDir" -ForegroundColor Yellow
    return
  }
  $cmakelistsPath = Join-Path $BuildDir 'CMakeLists.txt'
  Write-Section "Cleaning build cache under: $BuildDir (preserving CMakeLists.txt)"
  Get-ChildItem -LiteralPath $BuildDir -Force | ForEach-Object {
    if ($_.PSIsContainer) { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
    else { if (-not ($_.Name -ieq 'CMakeLists.txt')) { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue } }
  }
  Write-Host "[Clean] Completed." -ForegroundColor Green
}

function Configure-Project {
  param(
    [Parameter(Mandatory=$true)][string]$SourceDir,
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [switch]$Reconfigure,
    [string]$Generator = '',
    [string[]]$ExtraArgs = @()
  )
  if ($Reconfigure) {
    $cache = Join-Path $BuildDir 'CMakeCache.txt'
    $files = Join-Path $BuildDir 'CMakeFiles'
    if (Test-Path -LiteralPath $cache) { Remove-Item -LiteralPath $cache -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $files) { Remove-Item -LiteralPath $files -Recurse -Force -ErrorAction SilentlyContinue }
  }
  Write-Section "Configuring with CMake"
  $args = @('-S', $SourceDir, '-B', $BuildDir)
  if ($Generator) { $args = @('-G', $Generator) + $args }
  if ($ExtraArgs -and $ExtraArgs.Count -gt 0) { $args += $ExtraArgs }
  $configureCmd = @('cmake') + $args
  Write-Host "> $($configureCmd -join ' ')"
  & cmake @args
}

function Build-Project {
  param(
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][int]$Parallel
  )
  Write-Section "Building ($Config)"
  $buildCmd = @(
    'cmake',
    '--build', '"{0}"' -f $BuildDir,
    '--config', $Config,
    '--',
    "-j$Parallel"
  ) -join ' '
  Write-Host "> $buildCmd"
  & cmake --build $BuildDir --config $Config -- -j$Parallel
}

function Copy-ArtifactsToDir {
  <#
  .SYNOPSIS
    Copy built artifacts (DLL/EXE/LIB) to a specified directory.
  .PARAMETER BuildDir
    The directory to search recursively for artifacts.
  .PARAMETER DestDir
    Destination directory for artifacts.
  #>
  param(
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$DestDir
  )
  Write-Section "Collecting build artifacts to: $DestDir"
  Ensure-Directory -Path $DestDir
  $patterns = @('*.dll','*.exe','*.lib','*.pdb','*.exp','*.so','*.a')
  foreach ($pattern in $patterns) {
    Get-ChildItem -LiteralPath $BuildDir -Recurse -Include $pattern -File -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -notmatch '\\_deps\\' } |
      ForEach-Object {
        $dest = Join-Path $DestDir $_.Name
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
        Write-Host "[ART] " -NoNewline -ForegroundColor Yellow; Write-Host "$( $_.FullName ) -> $dest"
      }
  }
  Write-Host "[ART] Collection completed." -ForegroundColor Green
}

# ----------------------- Main -----------------------
$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Path
$RootDir   = Split-Path -Parent -Path $ScriptDir
$BuildDir  = Join-Path $RootDir 'build'
$ArtifactsDir = Join-Path $BuildDir 'Release'

Write-Section "Environment"
Write-Host ("RootDir : {0}" -f $RootDir)
Write-Host ("BuildDir: {0}" -f $BuildDir)
Write-Host ("ArtifactsDir: {0}" -f $ArtifactsDir)
Write-Host ("Config  : {0}" -f $Config)
Write-Host ("Parallel: {0}" -f $Parallel)
Write-Host ("Generator: {0}" -f $Generator)
Write-Host ("Toolchain: {0}" -f $Toolchain)
Write-Host ("EnableCuda: {0}" -f $EnableCuda)
Write-Host ("DisableTests: {0}" -f $DisableTests)
Write-Host ("DisableExamples: {0}" -f $DisableExamples)
Write-Host ("Visualization: {0}" -f $Visualization)
Write-Host ("AlsoLinux: {0}" -f $AlsoLinux)

Ensure-Directory -Path $BuildDir
Ensure-Directory -Path $ArtifactsDir

if ($Clean -or $CleanCache) {
  Clean-BuildCache -BuildDir $BuildDir
}

Configure-Project -SourceDir $RootDir -BuildDir $BuildDir -Reconfigure:$Reconfigure -Generator $Generator -ExtraArgs @(
  "-DCMAKE_BUILD_TYPE=$Config",
  (if ($DisableTests) { "-DBUILD_TESTS=OFF" } else { "-DBUILD_TESTS=ON" }),
  (if ($DisableExamples) { "-DBUILD_EXAMPLES=OFF" } else { "-DBUILD_EXAMPLES=ON" })
  ) + @(
    (if ($Visualization) { "-DBUILD_VISUALIZATION=ON" }),
    (if ($EnableCuda) { "-DENABLE_CUDA=ON" }),
    (if ($Toolchain) { "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" })
  )
Build-Project -BuildDir $BuildDir -Config $Config -Parallel $Parallel
Copy-ArtifactsToDir -BuildDir $BuildDir -DestDir $ArtifactsDir

if ($AlsoLinux) {
  Build-LinuxWithWSL -RootDir $RootDir -Config $Config -Parallel $Parallel -Generator $Generator -EnableCuda:$EnableCuda -DisableTests:$DisableTests -DisableExamples:$DisableExamples -Visualization:$Visualization
  $LinuxBuildDir = Join-Path $RootDir 'build-linux'
  Copy-ArtifactsToDir -BuildDir $LinuxBuildDir -DestDir $ArtifactsDir
}

Write-Section "Done"
Write-Host ("Build completed. Primary artifacts: {0}" -f (Join-Path $BuildDir $Config)) -ForegroundColor Green
Write-Host ("Convenience copies: {0}" -f $ArtifactsDir) -ForegroundColor Green

function Build-LinuxWithWSL {
  param(
    [Parameter(Mandatory=$true)][string]$RootDir,
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][int]$Parallel,
    [string]$Generator = '',
    [switch]$EnableCuda,
    [switch]$DisableTests,
    [switch]$DisableExamples,
    [switch]$Visualization
  )
  Write-Section "WSL Cross Build: Linux ($Config)"
  if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL 未安装或 wsl.exe 不可用，无法使用 -AlsoLinux"
  }
  $wslRoot = (& wsl.exe wslpath -a "$RootDir").Trim()
  if (-not $wslRoot) { throw "无法通过 wslpath 解析路径: $RootDir" }
  $wslBuild = "$wslRoot/build-linux"

  $args = @()
  if ($Generator) { $args += ("-G '" + $Generator + "'") }
  $args += "-S '$wslRoot'"
  $args += "-B '$wslBuild'"
  $args += "-DCMAKE_BUILD_TYPE=$Config"
  $args += (if ($DisableTests) { "-DBUILD_TESTS=OFF" } else { "-DBUILD_TESTS=ON" })
  $args += (if ($DisableExamples) { "-DBUILD_EXAMPLES=OFF" } else { "-DBUILD_EXAMPLES=ON" })
  if ($Visualization) { $args += "-DBUILD_VISUALIZATION=ON" }
  if ($EnableCuda) { $args += "-DENABLE_CUDA=ON" }

  $cfgCmd = "cmake $($args -join ' ')"
  $bldCmd = "cmake --build '$wslBuild' --config $Config -- -j$Parallel"
  $fullCmd = "$cfgCmd && $bldCmd"
  Write-Host "> (WSL) $fullCmd"
  & wsl.exe bash -lc "$fullCmd"
}