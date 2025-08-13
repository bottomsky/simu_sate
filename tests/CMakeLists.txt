cmake_minimum_required(VERSION 3.10)

# 测试总开关
option(BUILD_TESTING "Build unit/integration/performance tests" ON)

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