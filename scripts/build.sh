#!/bin/bash
# J2 轨道传播器构建脚本 (Linux/macOS)
# 支持清理缓存、配置构建选项和编译项目

set -e  # 遇到错误时退出

# 设置默认参数
BUILD_TYPE="Release"
CLEAN_BUILD=false
BUILD_EXAMPLES="ON"
BUILD_TESTS="ON"
BUILD_VISUALIZATION="OFF"
ENABLE_CUDA="OFF"
GENERATOR="Unix Makefiles"
VERBOSE=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
J2 轨道传播器构建脚本

用法: $0 [选项]

选项:
  --clean              清理构建缓存（保留CMakeLists.txt）
  --debug              构建Debug版本
  --release            构建Release版本（默认）
  --no-examples        不构建示例程序
  --no-tests           不构建测试程序
  --visualization      构建Vulkan可视化模块
  --cuda               强制启用CUDA支持
  --generator <gen>    指定CMake生成器（默认: "Unix Makefiles"）
  --jobs <num>         并行编译作业数（默认: 自动检测）
  --verbose            显示详细构建信息
  --help               显示此帮助信息

示例:
  $0 --clean --debug --visualization
  $0 --release --no-tests --cuda
  $0 --generator "Ninja" --jobs 8 --verbose

注意:
  - 构建可视化模块需要安装Vulkan SDK、GLFW和GLM
  - 启用CUDA需要安装CUDA Toolkit
  - 在macOS上可能需要安装Xcode命令行工具
  - 在Linux上可能需要安装build-essential包

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --visualization)
            BUILD_VISUALIZATION="ON"
            shift
            ;;
        --cuda)
            ENABLE_CUDA="ON"
            shift
            ;;
        --generator)
            GENERATOR="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_info "========================================"
print_info "J2 轨道传播器构建脚本"
print_info "========================================"
print_info "项目根目录: $(pwd)"
print_info "构建类型: $BUILD_TYPE"
print_info "生成器: $GENERATOR"
print_info "并行作业数: $JOBS"
print_info "清理构建: $CLEAN_BUILD"
print_info "构建示例: $BUILD_EXAMPLES"
print_info "构建测试: $BUILD_TESTS"
print_info "构建可视化: $BUILD_VISUALIZATION"
print_info "启用CUDA: $ENABLE_CUDA"
print_info "========================================"

# 检查CMake是否可用
if ! command -v cmake &> /dev/null; then
    print_error "未找到CMake，请确保CMake已安装并在PATH中"
    exit 1
fi

# 检查可视化模块依赖
if [[ "$BUILD_VISUALIZATION" == "ON" ]]; then
    print_info "检查可视化模块依赖..."
    
    # 检查Vulkan SDK
    if [[ -z "$VULKAN_SDK" ]] && ! command -v vulkaninfo &> /dev/null; then
        print_warning "未检测到Vulkan SDK，请确保已安装Vulkan SDK"
    fi
    
    # 检查pkg-config（用于查找GLFW和GLM）
    if command -v pkg-config &> /dev/null; then
        if ! pkg-config --exists glfw3; then
            print_warning "未检测到GLFW3，请确保已安装libglfw3-dev"
        fi
        if ! pkg-config --exists glm; then
            print_warning "未检测到GLM，请确保已安装libglm-dev"
        fi
    else
        print_warning "未找到pkg-config，无法检查GLFW和GLM依赖"
    fi
fi

# 检查CUDA依赖
if [[ "$ENABLE_CUDA" == "ON" ]]; then
    print_info "检查CUDA依赖..."
    if ! command -v nvcc &> /dev/null; then
        print_warning "未检测到CUDA编译器(nvcc)，请确保已安装CUDA Toolkit"
    fi
fi

# 创建构建目录
if [[ ! -d "build" ]]; then
    print_info "创建构建目录..."
    mkdir build
fi

# 清理构建缓存（保留CMakeLists.txt）
if [[ "$CLEAN_BUILD" == "true" ]]; then
    print_info "清理构建缓存..."
    cd build
    find . -mindepth 1 -maxdepth 1 ! -name "CMakeLists.txt" -exec rm -rf {} + 2>/dev/null || true
    cd ..
    print_success "构建缓存已清理"
fi

# 进入构建目录
cd build

# 配置CMake参数
CMAKE_ARGS=(
    -G "$GENERATOR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES"
    -DBUILD_TESTS="$BUILD_TESTS"
    -DBUILD_VISUALIZATION="$BUILD_VISUALIZATION"
    -DENABLE_CUDA="$ENABLE_CUDA"
)

# 运行CMake配置
print_info "配置项目..."
if [[ "$VERBOSE" == "true" ]]; then
    print_info "执行命令: cmake ${CMAKE_ARGS[*]} .."
fi

if ! cmake "${CMAKE_ARGS[@]}" ..; then
    print_error "CMake配置失败"
    exit 1
fi

# 构建项目
print_info "构建项目..."
BUILD_ARGS=(--build . --config "$BUILD_TYPE" --parallel "$JOBS")

if [[ "$VERBOSE" == "true" ]]; then
    BUILD_ARGS+=(--verbose)
    print_info "执行命令: cmake ${BUILD_ARGS[*]}"
fi

if ! cmake "${BUILD_ARGS[@]}"; then
    print_error "项目构建失败"
    exit 1
fi

print_success "========================================"
print_success "构建完成！"
print_success "========================================"
print_info "可执行文件位置: $(pwd)/$BUILD_TYPE"

if [[ "$BUILD_EXAMPLES" == "ON" ]] && [[ -f "$BUILD_TYPE/j2_example" ]]; then
    print_info "示例程序: j2_example"
fi

if [[ "$BUILD_VISUALIZATION" == "ON" ]] && [[ -f "$BUILD_TYPE/orbit_visualization_demo" ]]; then
    print_info "可视化演示: orbit_visualization_demo"
fi

print_success "========================================"

# 可选：运行测试
if [[ "$BUILD_TESTS" == "ON" ]]; then
    echo
    read -p "是否运行测试？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "运行测试..."
        if ! ctest --build-config "$BUILD_TYPE" --output-on-failure; then
            print_error "测试失败"
            exit 1
        fi
        print_success "所有测试通过！"
    fi
fi

# 显示系统信息（调试用）
if [[ "$VERBOSE" == "true" ]]; then
    echo
    print_info "系统信息:"
    print_info "操作系统: $(uname -s)"
    print_info "架构: $(uname -m)"
    print_info "CMake版本: $(cmake --version | head -n1)"
    if command -v gcc &> /dev/null; then
        print_info "GCC版本: $(gcc --version | head -n1)"
    fi
    if command -v clang &> /dev/null; then
        print_info "Clang版本: $(clang --version | head -n1)"
    fi
fi