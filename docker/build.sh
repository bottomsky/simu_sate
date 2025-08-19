#!/bin/bash

# Alpine Linux 构建脚本
# 使用 prantlf/alpine-make-gcc:latest 基础镜像构建 J2 Orbit Propagator

set -e  # 遇到错误时退出

# 颜色输出
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

# 检查 Docker 是否可用
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装或不在 PATH 中"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker 服务未运行"
        exit 1
    fi
    
    print_success "Docker 环境检查通过"
}

# 清理旧的镜像和容器
cleanup() {
    print_info "清理旧的 Docker 镜像和容器..."
    
    # 停止并删除相关容器
    docker ps -a --filter "ancestor=j2-orbit-propagator-alpine" --format "{{.ID}}" | xargs -r docker rm -f
    
    # 删除旧镜像
    if docker images | grep -q "j2-orbit-propagator-alpine"; then
        docker rmi j2-orbit-propagator-alpine:latest 2>/dev/null || true
    fi
    
    print_success "清理完成"
}

# 构建镜像
build_image() {
    print_info "开始构建 Alpine 版本的 J2 Orbit Propagator..."
    
    # 确保在项目根目录
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "请在项目根目录运行此脚本"
        exit 1
    fi
    
    # 构建镜像
    docker build \
        -f docker/Dockerfile.alpine \
        -t j2-orbit-propagator-alpine:latest \
        . || {
        print_error "Docker 构建失败"
        exit 1
    }
    
    print_success "镜像构建完成"
}

# 验证构建结果
verify_build() {
    print_info "验证构建结果..."
    
    # 运行容器并检查库文件
    docker run --rm j2-orbit-propagator-alpine:latest sh -c "
        echo '=== 验证库文件 ==='
        ls -la /usr/local/lib/
        echo '=== 验证动态链接 ==='
        ldd /usr/local/lib/*.so 2>/dev/null || echo '没有找到 .so 文件'
        echo '=== 验证完成 ==='
    " || {
        print_warning "验证过程中出现问题，但构建可能仍然成功"
    }
    
    print_success "构建验证完成"
}

# 提取构建产物
extract_artifacts() {
    print_info "提取构建产物到本地..."
    
    # 创建输出目录
    mkdir -p build/alpine-artifacts
    
    # 创建临时容器并复制文件
    container_id=$(docker create j2-orbit-propagator-alpine:latest)
    
    # 复制库文件
    docker cp "$container_id:/usr/local/lib/." build/alpine-artifacts/ 2>/dev/null || {
        print_warning "无法复制库文件，可能没有生成"
    }
    
    # 删除临时容器
    docker rm "$container_id" > /dev/null
    
    # 显示提取的文件
    if [ "$(ls -A build/alpine-artifacts 2>/dev/null)" ]; then
        print_success "构建产物已提取到 build/alpine-artifacts/"
        echo "提取的文件:"
        ls -la build/alpine-artifacts/
    else
        print_warning "没有找到构建产物"
    fi
}

# 显示使用说明
show_usage() {
    echo "Alpine Linux 构建脚本使用说明:"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --no-cleanup    跳过清理步骤"
    echo "  --no-extract    跳过提取构建产物"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 完整构建流程"
    echo "  $0 --no-cleanup      # 构建但不清理旧镜像"
    echo "  $0 --no-extract      # 构建但不提取产物"
}

# 主函数
main() {
    local skip_cleanup=false
    local skip_extract=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-cleanup)
                skip_cleanup=true
                shift
                ;;
            --no-extract)
                skip_extract=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "未知参数: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_info "开始 Alpine Linux 构建流程..."
    
    # 执行构建步骤
    check_docker
    
    if [ "$skip_cleanup" = false ]; then
        cleanup
    fi
    
    build_image
    verify_build
    
    if [ "$skip_extract" = false ]; then
        extract_artifacts
    fi
    
    print_success "Alpine Linux 构建流程完成!"
    print_info "镜像名称: j2-orbit-propagator-alpine:latest"
    
    if [ "$skip_extract" = false ]; then
        print_info "构建产物位置: build/alpine-artifacts/"
    fi
}

# 运行主函数
main "$@"