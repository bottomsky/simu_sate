#!/bin/bash

# J2 Orbit Propagator - Linux Build Script
# This script builds the J2 orbit propagator dynamic library on Linux

set -e  # Exit on any error

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
CLEAN_BUILD=false
INSTALL_BUILD=false
ENABLE_CUDA=false
ENABLE_TESTS=true
ENABLE_EXAMPLES=true
GENERATOR="Unix Makefiles"
JOBS=$(nproc)

# Function to display help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build the J2 Orbit Propagator dynamic library on Linux.

OPTIONS:
    -t, --type TYPE         Build type (Debug, Release, RelWithDebInfo, MinSizeRel)
                           Default: Release
    -b, --build-dir DIR    Build directory path
                           Default: build
    -c, --clean            Clean build directory before building
    -i, --install          Install after building
    -g, --generator GEN    CMake generator to use
                           Default: Unix Makefiles
    -j, --jobs NUM         Number of parallel jobs
                           Default: $(nproc)
    --enable-cuda          Force enable CUDA support
    --disable-tests        Disable building tests
    --disable-examples     Disable building examples
    -h, --help             Show this help message

EXAMPLES:
    $0                                    # Basic build
    $0 -t Debug -c                       # Clean debug build
    $0 -b /tmp/j2_build --enable-cuda    # Build with CUDA in custom directory
    $0 -t Release -i -j 8                # Release build with install using 8 jobs

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -i|--install)
            INSTALL_BUILD=true
            shift
            ;;
        -g|--generator)
            GENERATOR="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --enable-cuda)
            ENABLE_CUDA=true
            shift
            ;;
        --disable-tests)
            ENABLE_TESTS=false
            shift
            ;;
        --disable-examples)
            ENABLE_EXAMPLES=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate build type
case $BUILD_TYPE in
    Debug|Release|RelWithDebInfo|MinSizeRel)
        ;;
    *)
        echo "Error: Invalid build type '$BUILD_TYPE'"
        echo "Valid types: Debug, Release, RelWithDebInfo, MinSizeRel"
        exit 1
        ;;
esac

# Convert relative path to absolute
if [[ ! "$BUILD_DIR" = /* ]]; then
    BUILD_DIR="$(pwd)/$BUILD_DIR"
fi

echo "=== J2 Orbit Propagator Linux Build ==="
echo "Build type: $BUILD_TYPE"
echo "Build directory: $BUILD_DIR"
echo "Generator: $GENERATOR"
echo "Jobs: $JOBS"
echo "CUDA enabled: $ENABLE_CUDA"
echo "Tests enabled: $ENABLE_TESTS"
echo "Examples enabled: $ENABLE_EXAMPLES"
echo "Clean build: $CLEAN_BUILD"
echo "Install: $INSTALL_BUILD"
echo

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Configure CMake
echo "Configuring CMake..."
CMAKE_ARGS=(
    -G "$GENERATOR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DBUILD_TESTS="$ENABLE_TESTS"
    -DBUILD_EXAMPLES="$ENABLE_EXAMPLES"
)

if [ "$ENABLE_CUDA" = true ]; then
    CMAKE_ARGS+=(-DENABLE_CUDA=ON)
fi

if [ "$INSTALL_BUILD" = true ]; then
    CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install")
fi

cd "$BUILD_DIR"
cmake "${CMAKE_ARGS[@]}" ..

# Build
echo "Building..."
cmake --build . --config "$BUILD_TYPE" --parallel "$JOBS"

# Install if requested
if [ "$INSTALL_BUILD" = true ]; then
    echo "Installing..."
    cmake --install . --config "$BUILD_TYPE"
fi

# Copy shared library to examples directory if it exists
if [ "$ENABLE_EXAMPLES" = true ] && [ -f "libj2_orbit_propagator.so" ]; then
    EXAMPLE_DIR="../examples"
    if [ -d "$EXAMPLE_DIR" ]; then
        echo "Copying shared library to examples directory..."
        cp libj2_orbit_propagator.so "$EXAMPLE_DIR/"
    fi
fi

echo
echo "=== Build Summary ==="
echo "Build completed successfully!"
echo "Build directory: $BUILD_DIR"
echo "Build type: $BUILD_TYPE"

# List generated files
echo
echo "Generated files:"
find . -name "*.so" -o -name "*.a" -o -name "*_tests" -o -name "j2_example" | sort

echo
echo "To run tests: cd $BUILD_DIR && ctest --output-on-failure"
if [ "$ENABLE_EXAMPLES" = true ] && [ -f "j2_example" ]; then
    echo "To run example: cd $BUILD_DIR && ./j2_example"
fi