cmake_minimum_required(VERSION 3.10)

# 测试总开关
option(BUILD_TESTING "Build unit/integration/performance tests" ON)

# 新增：允许在 tests 子项目内强制启用/禁用 CUDA
option(ENABLE_CUDA "Force enable CUDA language for tests" OFF)
option(DISABLE_CUDA "Force disable CUDA language for tests" OFF)

# 当脚本以 tests 为顶层配置时，这里负责启用/禁用 CUDA 语言
if(DISABLE_CUDA)
    message(STATUS "[tests] CUDA explicitly disabled (DISABLE_CUDA=ON)")
elseif(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90")
    endif()
    message(STATUS "[tests] CUDA forced ON (ENABLE_CUDA=ON), arch: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90")
        endif()
        message(STATUS "[tests] CUDA detected and enabled, arch: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "[tests] CUDA not found - tests will run on CPU only")
    endif()
endif()

if(NOT BUILD_TESTING)
    return()
endif()

# 尝试查找GTest（优先使用系统/包管理器版本）
find_package(GTest QUIET)

# 如果找不到，则使用FetchContent自动拉取
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        # 使用Gitee镜像以提高下载速度和稳定性
        URL https://gitee.com/mirrors/googletest/repository/archive/v1.14.0.zip
        # 备用GitHub源
        # URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        TIMEOUT 30
    )
    # 禁止gtest强制覆盖父工程编译选项
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# 启用CTest
include(CTest)
if(BUILD_TESTING)
    enable_testing()
endif()

# 添加测试子目录
add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(performance)