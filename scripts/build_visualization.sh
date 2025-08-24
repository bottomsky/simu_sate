#!/bin/bash

# J2 轨道传播器 Vulkan 可视化构建脚本 (Linux 版本)
#
# 此脚本用于：
# 1. 检查 Vulkan SDK 是否已安装
# 2. 配置 CMake 并启用可视化模块
# 3. 编译项目
# 4. 运行可视化演示程序
#
# 用法:
#   ./build_visualization.sh [选项]
#
# 选项:
#   --clean         清理构建缓存（保留 CMakeLists.txt）
#   --release       使用 Release 构建类型（默认：Debug）
#   --skip-build    跳过编译，直接运行演示程序
#   --help          显示帮助信息

set -e  # 遇到错误时退出

# 默认参数
CLEAN=false
BUILD_TYPE="Debug"
SKIP_BUILD=false

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 输出函数
print_error() {
    echo -e "${RED}错误: $1${NC}" >&2
    exit 1
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_header() {
    echo -e "${MAGENTA}$1${NC}"
}

# 显示帮助信息
show_help() {
    echo "J2 轨道传播器 Vulkan 可视化构建脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --clean         清理构建缓存（保留 CMakeLists.txt）"
    echo "  --release       使用 Release 构建类型（默认：Debug）"
    echo "  --skip-build    跳过编译，直接运行演示程序"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0"
    echo "  $0 --clean --release"
    echo "  $0 --skip-build"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "未知参数: $1。使用 --help 查看帮助信息。"
            ;;
    esac
done

# 检查 Vulkan SDK
check_vulkan_sdk() {
    print_info "检查 Vulkan SDK..."
    
    # 检查环境变量
    if [[ -n "$VULKAN_SDK" ]]; then
        print_success "找到 Vulkan SDK: $VULKAN_SDK"
        
        # 验证 vulkaninfo 工具
        if [[ -f "$VULKAN_SDK/bin/vulkaninfo" ]]; then
            print_success "Vulkan SDK 验证通过"
            return 0
        else
            print_warning "Vulkan SDK 安装不完整，未找到 vulkaninfo"
        fi
    else
        print_warning "未找到 VULKAN_SDK 环境变量"
    fi
    
    # 尝试在系统中查找 vulkaninfo
    if command -v vulkaninfo >/dev/null 2>&1; then
        print_success "在系统路径中找到 vulkaninfo"
        return 0
    fi
    
    # 检查常见安装位置
    local common_paths=(
        "/usr/bin/vulkaninfo"
        "/usr/local/bin/vulkaninfo"
        "$HOME/VulkanSDK/*/x86_64/bin/vulkaninfo"
    )
    
    for path in "${common_paths[@]}"; do
        if [[ -f $path ]]; then
            print_success "在 $path 找到 vulkaninfo"
            return 0
        fi
    done
    
    print_error "未找到 Vulkan SDK。请安装 Vulkan SDK:\n" \
                "Ubuntu/Debian: sudo apt install vulkan-tools libvulkan-dev\n" \
                "Fedora/RHEL: sudo dnf install vulkan-tools vulkan-devel\n" \
                "Arch: sudo pacman -S vulkan-tools vulkan-headers\n" \
                "或从 https://vulkan.lunarg.com/ 下载官方 SDK"
}

# 检查必要工具
check_required_tools() {
    print_info "检查必要工具..."
    
    # 检查 CMake
    if command -v cmake >/dev/null 2>&1; then
        local cmake_version=$(cmake --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
        print_success "找到 CMake $cmake_version"
    else
        print_error "未找到 CMake。请安装 CMake 3.15 或更高版本"
    fi
    
    # 检查编译器
    if command -v g++ >/dev/null 2>&1; then
        local gcc_version=$(g++ --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
        print_success "找到 GCC $gcc_version"
    elif command -v clang++ >/dev/null 2>&1; then
        local clang_version=$(clang++ --version | head -n1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+')
        print_success "找到 Clang $clang_version"
    else
        print_error "未找到 C++ 编译器。请安装 GCC 或 Clang"
    fi
    
    # 检查 make 或 ninja
    if command -v make >/dev/null 2>&1; then
        print_success "找到 Make"
    elif command -v ninja >/dev/null 2>&1; then
        print_success "找到 Ninja"
    else
        print_warning "未找到 Make 或 Ninja，可能影响编译"
    fi
}

# 清理构建目录
clear_build_cache() {
    local build_dir="$1"
    
    print_info "清理构建缓存..."
    
    if [[ -d "$build_dir" ]]; then
        # 保存 CMakeLists.txt
        local cmake_lists="$build_dir/CMakeLists.txt"
        local temp_cmake="/tmp/cmake_lists_backup_$$"
        
        if [[ -f "$cmake_lists" ]]; then
            cp "$cmake_lists" "$temp_cmake"
        fi
        
        # 删除构建目录内容（除了 CMakeLists.txt）
        find "$build_dir" -mindepth 1 -not -name "CMakeLists.txt" -delete 2>/dev/null || true
        
        # 恢复 CMakeLists.txt
        if [[ -f "$temp_cmake" ]]; then
            cp "$temp_cmake" "$cmake_lists"
            rm "$temp_cmake"
        fi
        
        print_success "构建缓存已清理"
    else
        print_info "构建目录不存在，无需清理"
    fi
}

# 配置 CMake
configure_cmake() {
    local source_dir="$1"
    local build_dir="$2"
    local build_type="$3"
    
    print_info "配置 CMake..."
    
    # 确保构建目录存在
    mkdir -p "$build_dir"
    
    # 选择生成器
    local generator="Unix Makefiles"
    if command -v ninja >/dev/null 2>&1; then
        generator="Ninja"
    fi
    
    # CMake 配置参数
    local cmake_args=(
        "-G" "$generator"
        "-DCMAKE_BUILD_TYPE=$build_type"
        "-DBUILD_EXAMPLES=ON"
        "-DBUILD_TESTS=ON"
        "-DBUILD_VISUALIZATION=ON"
        "-DENABLE_CUDA=OFF"
        "$source_dir"
    )
    
    pushd "$build_dir" >/dev/null
    if cmake "${cmake_args[@]}"; then
        print_success "CMake 配置完成"
    else
        popd >/dev/null
        print_error "CMake 配置失败"
    fi
    popd >/dev/null
}

# 编译项目
build_project() {
    local build_dir="$1"
    local build_type="$2"
    
    print_info "编译项目..."
    
    pushd "$build_dir" >/dev/null
    if cmake --build . --config "$build_type" --target orbit_visualization_demo; then
        print_success "项目编译完成"
    else
        popd >/dev/null
        print_error "项目编译失败"
    fi
    popd >/dev/null
}

# 运行可视化演示
run_visualization_demo() {
    local build_dir="$1"
    local build_type="$2"
    
    print_info "启动可视化演示程序..."
    
    # 查找可执行文件
    local exe_paths=(
        "$build_dir/orbit_visualization_demo"
        "$build_dir/visualization/examples/orbit_visualization_demo"
        "$build_dir/bin/orbit_visualization_demo"
        "$build_dir/$build_type/orbit_visualization_demo"
    )
    
    local exe_path=""
    for path in "${exe_paths[@]}"; do
        if [[ -f "$path" && -x "$path" ]]; then
            exe_path="$path"
            break
        fi
    done
    
    if [[ -z "$exe_path" ]]; then
        print_error "未找到可视化演示程序可执行文件"
    fi
    
    print_success "找到可执行文件: $exe_path"
    print_info "正在启动可视化演示程序..."
    print_warning "按 ESC 键退出演示程序"
    
    if "$exe_path"; then
        print_success "演示程序正常退出"
    else
        local exit_code=$?
        print_warning "演示程序退出，退出码: $exit_code"
    fi
}

# 主函数
main() {
    print_header "=== J2 轨道传播器 Vulkan 可视化构建脚本 ==="
    echo
    
    # 获取项目根目录
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local project_root="$(dirname "$script_dir")"
    local build_dir="$project_root/build"
    
    print_info "项目根目录: $project_root"
    print_info "构建目录: $build_dir"
    print_info "构建类型: $BUILD_TYPE"
    echo
    
    # 检查 Vulkan SDK
    check_vulkan_sdk
    echo
    
    # 检查必要工具
    check_required_tools
    echo
    
    if [[ "$SKIP_BUILD" != "true" ]]; then
        # 清理构建缓存（如果需要）
        if [[ "$CLEAN" == "true" ]]; then
            clear_build_cache "$build_dir"
            echo
        fi
        
        # 配置 CMake
        configure_cmake "$project_root" "$build_dir" "$BUILD_TYPE"
        echo
        
        # 编译项目
        build_project "$build_dir" "$BUILD_TYPE"
        echo
    else
        print_info "跳过编译步骤"
        echo
    fi
    
    # 运行可视化演示
    run_visualization_demo "$build_dir" "$BUILD_TYPE"
    
    echo
    print_success "脚本执行完成"
}

# 执行主函数
main "$@"