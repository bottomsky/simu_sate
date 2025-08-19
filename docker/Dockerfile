# 基于 C++ 开发基础镜像构建
# 使用包含 cmake、git、pkgconfig、eigen-dev、linux-headers 的基础镜像
FROM cpp-dev-base:latest AS builder

# 设置工作目录
WORKDIR /workspace

# 验证基础镜像中的开发工具
RUN echo "=== 验证基础镜像中的开发工具 ===" && \
    gcc --version && \
    g++ --version && \
    cmake --version && \
    git --version && \
    pkg-config --version && \
    echo "=== 验证 Eigen 库 ===" && \
    find /usr -name "Eigen" -type d 2>/dev/null | head -3 && \
    echo "基础镜像验证完成"

# 设置编译选项（仅针对C++源文件，避免影响CUDA编译）
RUN echo "=== Setting up build environment ===" && \
    export CXXFLAGS="-Wall -Wextra -pedantic -mavx2 -mfma -Wno-unused-variable -Wno-unused-but-set-variable" && \
    echo "CXXFLAGS set to: $CXXFLAGS"

# 复制源代码（从项目根目录复制）
COPY ../ .

# 完全清理任何现有的构建产物
RUN echo "=== Cleaning all build artifacts ===" && \
    rm -rf build/ && \
    rm -rf CMakeCache.txt && \
    rm -rf CMakeFiles/ && \
    find . -name "*.o" -delete && \
    find . -name "*.so" -delete && \
    find . -name "*.a" -delete && \
    echo "Cleanup completed"

# 验证源代码结构
RUN echo "=== Verifying source structure ===" && \
    ls -la && \
    echo "CMakeLists.txt exists:" && ls -la CMakeLists.txt && \
    echo "Include directory:" && ls -la include/ && \
    echo "Source directory:" && ls -la src/ && \
    echo "Header file check:" && ls -la include/constellation_propagator_c.h

# 验证编译环境
RUN echo "=== Verifying build environment ===" && \
    gcc --version && \
    g++ --version && \
    cmake --version && \
    echo "Environment verification completed"

# 创建新的构建目录
RUN echo "=== Creating fresh build directory ===" && \
    mkdir -p build && \
    cd build && \
    echo "Build directory created"

# 配置项目（Alpine 特定配置）
RUN echo "=== Configuring project for Alpine ===" && \
    cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_CUDA=OFF \
        -DENABLE_TESTS=OFF \
        -DENABLE_EXAMPLES=OFF \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_CXX_FLAGS="-fPIC -O3" \
        -DCMAKE_C_FLAGS="-fPIC -O3" && \
    echo "Configuration completed" && \
    echo "Generated files:" && ls -la

# 构建项目
RUN echo "=== Building project ===" && \
    cd build && \
    echo "Checking size_t definition:" && \
    echo '#include <cstddef>' > size_t_check.cpp && \
    echo '#include <iostream>' >> size_t_check.cpp && \
    echo 'int main() { std::cout << "size_t size: " << sizeof(size_t) << std::endl; return 0; }' >> size_t_check.cpp && \
    g++ -o size_t_check size_t_check.cpp && \
    ./size_t_check && \
    echo "Starting compilation with verbose output..." && \
    make -j$(nproc) VERBOSE=1 2>&1 | tee build.log || (echo "Build failed, showing error context:" && tail -50 build.log && false) && \
    echo "Build successful" && \
    echo "Build artifacts:" && ls -la && \
    echo "=== 动态库文件路径信息 ===" && \
    echo "容器内构建目录: /workspace/build" && \
    echo "当前工作目录: $(pwd)" && \
    echo "查找所有库文件:" && \
    find /workspace -name "*.so" -o -name "*.a" | while read file; do \
        echo "  库文件: $file" && \
        echo "    大小: $(stat -c%s "$file") bytes" && \
        echo "    修改时间: $(stat -c%y "$file")" \
    ; done && \
    echo "=== 库文件详细信息 ===" && \
    find . -name "*.so" -o -name "*.a" -exec ls -la {} \;

# 复制构建产物到输出目录
RUN echo "=== Copying build artifacts ===" && \
    mkdir -p /output && \
    echo "=== 复制动态库到输出目录 ===" && \
    find bin -name "*.so" -exec cp -v {} /output/ \; && \
    find bin -name "*.a" -exec cp -v {} /output/ \; && \
    echo "=== 输出目录信息 ===" && \
    echo "容器内输出目录: /output" && \
    echo "输出目录内容:" && \
    ls -la /output/ && \
    echo "=== 动态库挂载信息 ===" && \
    echo "要挂载的容器路径: /output" && \
    echo "建议的主机挂载路径: ./bin" && \
    echo "Docker 挂载命令示例: -v \$(pwd)/bin:/output" && \
    echo "=== 库文件完整路径列表 ===" && \
    find /output -type f | while read file; do \
        echo "  $file ($(stat -c%s "$file") bytes)" \
    ; done

# 最终验证
RUN echo "=== Final verification ===" && \
    echo "Workspace contents:" && ls -la && \
    echo "Build directory:" && ls -la build/ && \
    echo "Output directory:" && ls -la /output/ && \
    echo "Build completed successfully"

# 最小运行时镜像（基于 Alpine）
FROM alpine:latest AS runtime

# 安装运行时依赖
RUN apk add --no-cache \
        libstdc++ \
        eigen-dev && \
    echo "Runtime dependencies installed" && \
    echo "=== 运行时镜像信息 ===" && \
    echo "基础镜像: alpine:latest" && \
    echo "运行时依赖: libstdc++, eigen-dev"

# 复制构建产物
COPY --from=builder /output/* /usr/local/lib/

# 更新动态链接器缓存
RUN ldconfig /usr/local/lib

# 设置默认命令
CMD ["echo", "J2 Orbit Propagator library ready (Alpine Linux)"]