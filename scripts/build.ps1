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

.PARAMETER Clean
  If specified, removes all contents in the build directory except build/CMakeLists.txt.

.PARAMETER CleanCache
  Alias for cleaning build cache while preserving build/CMakeLists.txt (same effect as -Clean).

.PARAMETER Config
  The build configuration. Typical values: Debug or Release. Default is Release.

.PARAMETER Parallel
  Number of parallel build jobs. Defaults to number of logical processors.

.PARAMETER Reconfigure
  If specified, forces CMake to re-configure (delete CMakeCache.txt and CMakeFiles/ before configuring).

.EXAMPLE
  ./build.ps1 -Clean -Config Release

.EXAMPLE
  ./build.ps1 -CleanCache -Config Release

.EXAMPLE
  ./build.ps1 -Config Debug -Parallel 8

.NOTES
  - Requires CMake to be installed and available on PATH.
  - Supports both single-config and multi-config generators on Windows.
#>

[CmdletBinding()]
param(
  [switch]$Clean,
  [switch]$CleanCache,
  [string]$Config = 'Release',
  [int]$Parallel = [Environment]::ProcessorCount,
  [switch]$Reconfigure
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
  <#
  .SYNOPSIS
    Clean build directory contents while preserving build/CMakeLists.txt.
  .DESCRIPTION
    Removes all files and subdirectories under the build folder except the top-level
    CMakeLists.txt file. If the build folder does not exist, this step is skipped.
  .OUTPUTS
    None.
  .NOTES
    This operation is irreversible for deleted files. It does NOT delete build/CMakeLists.txt.
  #>
  param([string]$BuildDir)

  if (-not (Test-Path -LiteralPath $BuildDir)) {
    Write-Host "[Clean] Build directory not found, skipping clean: $BuildDir" -ForegroundColor Yellow
    return
  }

  $cmakelistsPath = Join-Path $BuildDir 'CMakeLists.txt'
  Write-Section "Cleaning build cache under: $BuildDir (preserving CMakeLists.txt)"

  # Delete everything at the top level except CMakeLists.txt
  Get-ChildItem -LiteralPath $BuildDir -Force | ForEach-Object {
    if ($_.PSIsContainer) {
      Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
    } else {
      if (-not ($_.Name -ieq 'CMakeLists.txt')) {
        Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
      }
    }
  }

  Write-Host "[Clean] Completed." -ForegroundColor Green
}

function Configure-Project {
  <#
  .SYNOPSIS
    Run CMake configure step (-S/-B) for the project.
  .PARAMETER SourceDir
    The project source directory containing the top-level CMakeLists.txt.
  .PARAMETER BuildDir
    The build directory where CMake will generate build files.
  .PARAMETER Reconfigure
    If set, remove CMakeCache.txt and CMakeFiles before configuring to force a fresh configure.
  #>
  param(
    [Parameter(Mandatory=$true)][string]$SourceDir,
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [switch]$Reconfigure
  )

  if ($Reconfigure) {
    $cache = Join-Path $BuildDir 'CMakeCache.txt'
    $files = Join-Path $BuildDir 'CMakeFiles'
    if (Test-Path -LiteralPath $cache) { Remove-Item -LiteralPath $cache -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $files) { Remove-Item -LiteralPath $files -Recurse -Force -ErrorAction SilentlyContinue }
  }

  Write-Section "Configuring with CMake"
  $configureCmd = @(
    'cmake',
    '-S', '"{0}"' -f $SourceDir,
    '-B', '"{0}"' -f $BuildDir
  ) -join ' '

  Write-Host "> $configureCmd"
  & cmake -S $SourceDir -B $BuildDir
}

function Build-Project {
  <#
  .SYNOPSIS
    Build the project using cmake --build.
  .PARAMETER BuildDir
    The build directory to build.
  .PARAMETER Config
    Build configuration (Release/Debug, etc.).
  .PARAMETER Parallel
    Parallel job count for the build.
  #>
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

function Copy-ArtifactsToBin {
  <#
  .SYNOPSIS
    Copy built artifacts (DLL/EXE) to the bin directory.
  .PARAMETER BuildDir
    The directory to search recursively for artifacts.
  .PARAMETER BinDir
    Destination directory for artifacts.
  #>
  param(
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$BinDir
  )

  Write-Section "Collecting build artifacts to: $BinDir"
  Ensure-Directory -Path $BinDir

  $patterns = @('*.dll','*.exe')
  foreach ($pattern in $patterns) {
    Get-ChildItem -LiteralPath $BuildDir -Recurse -Include $pattern -File -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -notmatch '\\_deps\\' } |
      ForEach-Object {
        $dest = Join-Path $BinDir $_.Name
        Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
        Write-Host "[BIN] " -NoNewline -ForegroundColor Yellow; Write-Host "$($_.FullName) -> $dest"
      }
  }

  Write-Host "[BIN] Collection completed." -ForegroundColor Green
}

# ----------------------- Main -----------------------
$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Path
$RootDir   = Split-Path -Parent -Path $ScriptDir
$BuildDir  = Join-Path $RootDir 'build'
$BinDir    = Join-Path $RootDir 'bin'

Write-Section "Environment"
Write-Host ("RootDir : {0}" -f $RootDir)
Write-Host ("BuildDir: {0}" -f $BuildDir)
Write-Host ("BinDir  : {0}" -f $BinDir)
Write-Host ("Config  : {0}" -f $Config)
Write-Host ("Parallel: {0}" -f $Parallel)

Ensure-Directory -Path $BuildDir
Ensure-Directory -Path $BinDir

if ($Clean -or $CleanCache) {
  Clean-BuildCache -BuildDir $BuildDir
}

Configure-Project -SourceDir $RootDir -BuildDir $BuildDir -Reconfigure:$Reconfigure
Build-Project -BuildDir $BuildDir -Config $Config -Parallel $Parallel
Copy-ArtifactsToBin -BuildDir $BuildDir -BinDir $BinDir

Write-Section "Done"
Write-Host ("Build completed. Primary artifacts: {0}" -f (Join-Path $BuildDir $Config)) -ForegroundColor Green
Write-Host ("Convenience copies (Windows DLL/EXE): {0}" -f $BinDir) -ForegroundColor Green