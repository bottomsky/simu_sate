cmake_minimum_required(VERSION 3.12)

# 新增：允许通过开关强制启用 CUDA 语言
option(ENABLE_CUDA "Force enable CUDA language" OFF)

# 检查/启用 CUDA 语言与工具包
if(ENABLE_CUDA)
    project(J2OrbitPropagator VERSION 1.0.0 LANGUAGES CXX CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA forced ON (ENABLE_CUDA=ON)")
    # 设置CUDA架构
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90")
        message(STATUS "CUDA architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        project(J2OrbitPropagator VERSION 1.0.0 LANGUAGES CXX CUDA)
        message(STATUS "CUDA Toolkit found, enabling CUDA language")
        # 设置CUDA架构
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90")
            message(STATUS "CUDA architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    else()
        project(J2OrbitPropagator VERSION 1.0.0 LANGUAGES CXX)
        message(STATUS "CUDA Toolkit not found")
    endif()
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录 - 所有编译产物统一输出到 bin 目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# 为不同配置类型设置输出目录
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_CURRENT_SOURCE_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_CURRENT_SOURCE_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_CURRENT_SOURCE_DIR}/bin)
endforeach()

# 设置编译选项（仅针对C++源文件，避免影响CUDA编译）
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra> $<$<COMPILE_LANGUAGE:CXX>:-pedantic> $<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma> $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable> $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-but-set-variable>)
endif()

# 使用项目中的Eigen库
set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen
)

# 源文件
set(SOURCES
    src/j2_orbit_propagator.cpp
    src/j2_orbit_propagator_c.cpp
    src/constellation_propagator.cpp
    src/constellation_propagator_c.cpp
)

# 头文件
set(HEADERS
    include/j2_orbit_propagator.h
    include/j2_orbit_propagator_c.h
    include/constellation_propagator.h
    include/constellation_propagator_c.h
    include/common_types.h
    include/math_defs.h
)

# 创建静态库
add_library(j2_orbit_propagator_static STATIC ${SOURCES} ${HEADERS})
target_link_libraries(j2_orbit_propagator_static Eigen3::Eigen)
target_include_directories(j2_orbit_propagator_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(j2_orbit_propagator_static PRIVATE J2_BUILD_STATIC)

# 创建动态库
add_library(j2_orbit_propagator_shared SHARED ${SOURCES} ${HEADERS})
target_link_libraries(j2_orbit_propagator_shared Eigen3::Eigen)
target_include_directories(j2_orbit_propagator_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 设置共享库目标属性
set_target_properties(j2_orbit_propagator_shared PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME j2_orbit_propagator
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# 为共享库定义导出宏
target_compile_definitions(j2_orbit_propagator_shared PRIVATE J2_BUILD_DLL)

# 可选：构建示例程序
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    # 检查是否存在main.cpp
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
        add_executable(j2_example src/main.cpp)
        target_link_libraries(j2_example j2_orbit_propagator_static)
    endif()
endif()

# 可选：构建测试
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 安装配置
install(TARGETS j2_orbit_propagator_static j2_orbit_propagator_shared
    EXPORT J2OrbitPropagatorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(FILES ${HEADERS}
    DESTINATION include
)

# 导出配置
install(EXPORT J2OrbitPropagatorTargets
    FILE J2OrbitPropagatorTargets.cmake
    NAMESPACE J2OrbitPropagator::
    DESTINATION lib/cmake/J2OrbitPropagator
)

# 创建配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "J2OrbitPropagatorConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/J2OrbitPropagatorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/J2OrbitPropagatorConfig.cmake"
    INSTALL_DESTINATION lib/cmake/J2OrbitPropagator
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/J2OrbitPropagatorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/J2OrbitPropagatorConfigVersion.cmake"
    DESTINATION lib/cmake/J2OrbitPropagator
)

# 打印构建信息
message(STATUS "J2 Orbit Propagator Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
if("CUDA" IN_LIST CMAKE_ENABLED_LANGUAGES)
    message(STATUS "  CUDA enabled via languages: ${CMAKE_ENABLED_LANGUAGES}")
endif()
if(CUDAToolkit_FOUND)
    message(STATUS "  CUDA version: ${CUDAToolkit_VERSION}")
endif()