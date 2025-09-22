#!/usr/bin/env bash
set -euo pipefail

# Configure and build the project on Linux/macOS, with unified artifact collection.
# Features:
#  - --clean: clean build cache (preserve build/CMakeLists.txt if any project requires it)
#  - --debug/--release: configuration (default Release)
#  - --jobs N: parallel build jobs
#  - --generator NAME: CMake generator (e.g., Ninja, Unix Makefiles)
#  - --toolchain <path|mingw|aarch64>: toolchain file or keyword
#  - --cuda: enable CUDA
#  - --no-tests / --no-examples: disable building tests/examples
#  - --visualization: build visualization module
#  - --also-windows: additionally cross-compile Windows using MinGW toolchain and collect artifacts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
WIN_BUILD_DIR="$ROOT_DIR/build-windows"
ART_DIR="$ROOT_DIR/build/Release"

CONFIG="Release"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
GENERATOR=""
TOOLCHAIN=""
ENABLE_CUDA=0
DISABLE_TESTS=0
DISABLE_EXAMPLES=0
VISUALIZATION=0
CLEAN=0
ALSO_WINDOWS=0

print_help() {
  cat <<EOF
Usage: bash scripts/build.sh [options]

Options:
  --clean                       Clean build cache
  --debug | --release           Build type (default: Release)
  --jobs N                      Parallel build jobs (default: autodetect)
  --generator NAME              CMake generator (e.g., Ninja, Unix Makefiles)
  --toolchain <path|mingw|aarch64>
                                Use CMake toolchain file or keyword
  --cuda                        Enable CUDA
  --no-tests                    Disable tests
  --no-examples                 Disable examples
  --visualization               Build visualization module
  --also-windows                Additionally cross-compile Windows target (MinGW) and collect artifacts
  -h | --help                   Show this help

Examples:
  bash scripts/build.sh --release --generator "Ninja" --jobs 8
  bash scripts/build.sh --release --no-tests --no-examples
  bash scripts/build.sh --release --visualization
  bash scripts/build.sh --release --toolchain aarch64
  bash scripts/build.sh --release --toolchain mingw
  bash scripts/build.sh --release --also-windows
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    --debug) CONFIG="Debug"; shift ;;
    --release) CONFIG="Release"; shift ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --generator) GENERATOR="$2"; shift 2 ;;
    --toolchain) TOOLCHAIN="$2"; shift 2 ;;
    --cuda) ENABLE_CUDA=1; shift ;;
    --no-tests) DISABLE_TESTS=1; shift ;;
    --no-examples) DISABLE_EXAMPLES=1; shift ;;
    --visualization) VISUALIZATION=1; shift ;;
    --also-windows) ALSO_WINDOWS=1; shift ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; print_help; exit 1 ;;
  esac
done

map_toolchain() {
  local key="$1"
  case "$key" in
    "") echo "" ;;
    mingw) echo "$ROOT_DIR/cmake/toolchains/windows-mingw.cmake" ;;
    aarch64) echo "$ROOT_DIR/cmake/toolchains/linux-aarch64.cmake" ;;
    *) echo "$key" ;;
  esac
}

TOOLCHAIN_FILE="$(map_toolchain "$TOOLCHAIN")"

section() { echo -e "\n==== $* ====\n"; }

section "Environment"
echo "RootDir      : $ROOT_DIR"
echo "BuildDir     : $BUILD_DIR"
echo "ArtifactsDir : $ART_DIR"
echo "Config       : $CONFIG"
echo "Jobs         : $JOBS"
echo "Generator    : ${GENERATOR:-<default>}"
echo "Toolchain    : ${TOOLCHAIN_FILE:-<none>}"
echo "CUDA         : $ENABLE_CUDA"
echo "NoTests      : $DISABLE_TESTS"
echo "NoExamples   : $DISABLE_EXAMPLES"
echo "Visualization: $VISUALIZATION"
echo "AlsoWindows  : $ALSO_WINDOWS"

mkdir -p "$BUILD_DIR" "$ART_DIR"

if [[ $CLEAN -eq 1 ]]; then
  section "Cleaning build cache: $BUILD_DIR"
  if [[ -d "$BUILD_DIR" ]]; then
    find "$BUILD_DIR" -mindepth 1 -maxdepth 1 -not -name 'CMakeLists.txt' -exec rm -rf {} + || true
  fi
fi

# Configure
section "Configuring with CMake"
CFG_ARGS=( -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CONFIG" )
if [[ -n "$GENERATOR" ]]; then CFG_ARGS=( -G "$GENERATOR" "${CFG_ARGS[@]}" ); fi
if [[ $DISABLE_TESTS -eq 1 ]]; then CFG_ARGS+=( -DBUILD_TESTS=OFF ); else CFG_ARGS+=( -DBUILD_TESTS=ON ); fi
if [[ $DISABLE_EXAMPLES -eq 1 ]]; then CFG_ARGS+=( -DBUILD_EXAMPLES=OFF ); else CFG_ARGS+=( -DBUILD_EXAMPLES=ON ); fi
if [[ $VISUALIZATION -eq 1 ]]; then CFG_ARGS+=( -DBUILD_VISUALIZATION=ON ); fi
if [[ $ENABLE_CUDA -eq 1 ]]; then CFG_ARGS+=( -DENABLE_CUDA=ON ); fi
if [[ -n "$TOOLCHAIN_FILE" ]]; then CFG_ARGS+=( -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" ); fi

echo "> cmake ${CFG_ARGS[*]}"
cmake "${CFG_ARGS[@]}"

# Build
section "Building ($CONFIG)"
echo "> cmake --build \"$BUILD_DIR\" --config $CONFIG -- -j$JOBS"
cmake --build "$BUILD_DIR" --config "$CONFIG" -- -j"$JOBS"

# Collect artifacts (Linux/macOS)
section "Collecting artifacts -> $ART_DIR"
# Shared/static libraries
find "$BUILD_DIR" -type f \( -name "*.so" -o -name "*.a" \) -not -path "*/_deps/*" -exec cp -f {} "$ART_DIR/" \; || true
# Executables ( +x files )
find "$BUILD_DIR" -type f -perm -111 -not -path "*/_deps/*" -exec cp -f {} "$ART_DIR/" \; || true

# Backward compatibility for examples using libj2_orbit_propagator.so
if [[ -f "$ART_DIR/libj2_orbit_propagator.so" ]]; then
  cp -f "$ART_DIR/libj2_orbit_propagator.so" "$ROOT_DIR/example/" || true
fi

# Optionally cross-compile Windows
if [[ $ALSO_WINDOWS -eq 1 ]]; then
  section "Cross-compiling Windows (MinGW)"
  WIN_TOOLCHAIN="$ROOT_DIR/cmake/toolchains/windows-mingw.cmake"
  if [[ ! -f "$WIN_TOOLCHAIN" ]]; then
    echo "Missing toolchain: $WIN_TOOLCHAIN" >&2
    exit 2
  fi
  mkdir -p "$WIN_BUILD_DIR"
  WIN_CFG_ARGS=( -S "$ROOT_DIR" -B "$WIN_BUILD_DIR" -DCMAKE_BUILD_TYPE="$CONFIG" -DCMAKE_TOOLCHAIN_FILE="$WIN_TOOLCHAIN" )
  if [[ -n "$GENERATOR" ]]; then WIN_CFG_ARGS=( -G "$GENERATOR" "${WIN_CFG_ARGS[@]}" ); fi
  if [[ $DISABLE_TESTS -eq 1 ]]; then WIN_CFG_ARGS+=( -DBUILD_TESTS=OFF ); else WIN_CFG_ARGS+=( -DBUILD_TESTS=ON ); fi
  if [[ $DISABLE_EXAMPLES -eq 1 ]]; then WIN_CFG_ARGS+=( -DBUILD_EXAMPLES=OFF ); else WIN_CFG_ARGS+=( -DBUILD_EXAMPLES=ON ); fi
  if [[ $VISUALIZATION -eq 1 ]]; then WIN_CFG_ARGS+=( -DBUILD_VISUALIZATION=ON ); fi
  if [[ $ENABLE_CUDA -eq 1 ]]; then WIN_CFG_ARGS+=( -DENABLE_CUDA=ON ); fi
  echo "> cmake ${WIN_CFG_ARGS[*]}"
  cmake "${WIN_CFG_ARGS[@]}"
  echo "> cmake --build \"$WIN_BUILD_DIR\" --config $CONFIG -- -j$JOBS"
  cmake --build "$WIN_BUILD_DIR" --config "$CONFIG" -- -j"$JOBS"
  section "Collecting Windows artifacts -> $ART_DIR"
  find "$WIN_BUILD_DIR" -type f \( -name "*.dll" -o -name "*.exe" -o -name "*.lib" \) -not -path "*/_deps/*" -exec cp -f {} "$ART_DIR/" \; || true
fi

section "Done"
echo "Primary artifacts (per-config): $BUILD_DIR/$CONFIG"
echo "Convenience copies            : $ART_DIR"