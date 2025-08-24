@echo off
setlocal enabledelayedexpansion

:: J2 Orbit Propagator Build Script (Windows)
:: Supports cache cleaning, build configuration and project compilation

:: Set default parameters
set BUILD_TYPE=Release
set CLEAN_BUILD=false
set BUILD_EXAMPLES=ON
set BUILD_TESTS=ON
set BUILD_VISUALIZATION=OFF
set ENABLE_CUDA=OFF
set GENERATOR=Visual Studio 17 2022
set PLATFORM=x64
set VERBOSE=false
set JOBS=%NUMBER_OF_PROCESSORS%

:: Get project root directory
for %%i in ("%~dp0..") do set PROJECT_ROOT=%%~fi
cd /d "%PROJECT_ROOT%"

:: Parse command line arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--clean" (
    set CLEAN_BUILD=true
    shift
    goto parse_args
)
if "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto parse_args
)
if "%~1"=="--release" (
    set BUILD_TYPE=Release
    shift
    goto parse_args
)
if "%~1"=="--no-examples" (
    set BUILD_EXAMPLES=OFF
    shift
    goto parse_args
)
if "%~1"=="--no-tests" (
    set BUILD_TESTS=OFF
    shift
    goto parse_args
)
if "%~1"=="--visualization" (
    set BUILD_VISUALIZATION=ON
    shift
    goto parse_args
)
if "%~1"=="--cuda" (
    set ENABLE_CUDA=ON
    shift
    goto parse_args
)
if "%~1"=="--generator" (
    set GENERATOR=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--platform" (
    set PLATFORM=%~2
    shift
    shift
    goto parse_args
)
if "%~1"=="--verbose" (
    set VERBOSE=true
    shift
    goto parse_args
)
if "%~1"=="--help" (
    goto show_help
)
echo Error: Unknown parameter %~1
goto show_help

:end_parse

echo ========================================
echo J2 Orbit Propagator Build Script
echo ========================================
echo Project root: %PROJECT_ROOT%
echo Build type: %BUILD_TYPE%
echo Generator: %GENERATOR%
echo Platform: %PLATFORM%
echo Jobs: %JOBS%
echo Clean build: %CLEAN_BUILD%
echo Build examples: %BUILD_EXAMPLES%
echo Build tests: %BUILD_TESTS%
echo Build visualization: %BUILD_VISUALIZATION%
echo Enable CUDA: %ENABLE_CUDA%
echo ========================================

:: Check if CMake is available
cmake --version >nul 2>&1
if errorlevel 1 (
    echo Error: CMake not found. Please install CMake and add it to PATH.
    exit /b 1
)

:: Check visualization dependencies
if "%BUILD_VISUALIZATION%"=="ON" (
    echo Checking visualization dependencies...
    if "%VULKAN_SDK%"=="" (
        echo Warning: VULKAN_SDK environment variable not set.
        echo Please install Vulkan SDK and set VULKAN_SDK environment variable.
    ) else (
        echo Found Vulkan SDK: %VULKAN_SDK%
    )
)

:: Check CUDA dependencies
if "%ENABLE_CUDA%"=="ON" (
    echo Checking CUDA dependencies...
    nvcc --version >nul 2>&1
    if errorlevel 1 (
        echo Warning: CUDA compiler (nvcc) not found.
        echo Please install CUDA Toolkit and add it to PATH.
    ) else (
        echo Found CUDA compiler
    )
)

:: Create build directory
if exist "build" goto skip_mkdir
echo Creating build directory...
mkdir build
:skip_mkdir

:: Clean build cache (preserve CMakeLists.txt)
if "%CLEAN_BUILD%"=="true" (
    echo Cleaning build cache...
    cd build
    for /f "delims=" %%i in ('dir /b /a-d 2^>nul ^| findstr /v CMakeLists.txt') do del "%%i" >nul 2>&1
    for /f "delims=" %%i in ('dir /b /ad 2^>nul') do rmdir /s /q "%%i" >nul 2>&1
    cd ..
    echo Build cache cleaned
)

:: Enter build directory
cd build

:: Configure CMake arguments
set CMAKE_ARGS=-G "%GENERATOR%" -A "%PLATFORM%" -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" -DBUILD_EXAMPLES="%BUILD_EXAMPLES%" -DBUILD_TESTS="%BUILD_TESTS%" -DBUILD_VISUALIZATION="%BUILD_VISUALIZATION%" -DENABLE_CUDA="%ENABLE_CUDA%"

:: Run CMake configuration
echo Configuring project...
if "%VERBOSE%"=="true" (
    echo Executing: cmake %CMAKE_ARGS% ..
)

cmake %CMAKE_ARGS% ..
if errorlevel 1 (
    echo Error: CMake configuration failed
    exit /b 1
)

:: Build project
echo Building project...
set BUILD_ARGS=--build . --config "%BUILD_TYPE%" --parallel "%JOBS%"

if "%VERBOSE%"=="true" (
    set BUILD_ARGS=%BUILD_ARGS% --verbose
    echo Executing: cmake %BUILD_ARGS%
)

cmake %BUILD_ARGS%
if errorlevel 1 (
    echo Error: Project build failed
    exit /b 1
)

echo ========================================
echo Build completed successfully!
echo ========================================
echo Executable location: %cd%\%BUILD_TYPE%

if "%BUILD_EXAMPLES%"=="ON" (
    if exist "%BUILD_TYPE%\j2_example.exe" (
        echo Example program: j2_example.exe
    )
)

if "%BUILD_VISUALIZATION%"=="ON" (
    if exist "%BUILD_TYPE%\orbit_visualization_demo.exe" (
        echo Visualization demo: orbit_visualization_demo.exe
    )
)

echo ========================================

:: Optional: Run tests
if "%BUILD_TESTS%"=="ON" (
    echo.
    set /p run_tests=Run tests? (y/N): 
    if /i "!run_tests!"=="y" (
        echo Running tests...
        ctest --build-config "%BUILD_TYPE%" --output-on-failure
        if errorlevel 1 (
            echo Error: Tests failed
            exit /b 1
        )
        echo All tests passed!
    )
)

:: Display system information (for debugging)
if "%VERBOSE%"=="true" (
    echo.
    echo System information:
    echo OS: %OS%
    echo Processor: %PROCESSOR_ARCHITECTURE%
    cmake --version | findstr "cmake version"
    if exist "%ProgramFiles%\Microsoft Visual Studio" (
        echo Visual Studio: Installed
    )
)

goto end

:show_help
echo J2 Orbit Propagator Build Script
echo.
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --clean              Clean build cache (preserve CMakeLists.txt)
echo   --debug              Build Debug version
echo   --release            Build Release version (default)
echo   --no-examples        Do not build example programs
echo   --no-tests           Do not build test programs
echo   --visualization      Build Vulkan visualization module
echo   --cuda               Force enable CUDA support
echo   --generator ^<gen^>     Specify CMake generator (default: "Visual Studio 17 2022")
echo   --platform ^<plat^>     Specify platform (default: x64)
echo   --verbose            Show detailed build information
echo   --help               Show this help information
echo.
echo Examples:
echo   %~nx0 --clean --debug --visualization
echo   %~nx0 --release --no-tests --cuda
echo   %~nx0 --generator "Visual Studio 16 2019" --platform Win32
echo.
echo Notes:
echo   - Building visualization module requires Vulkan SDK, GLFW, and GLM
echo   - Enabling CUDA requires CUDA Toolkit installation
echo   - Visual Studio 2019 or later is recommended
echo   - vcpkg can be used to install dependencies
echo.

:end