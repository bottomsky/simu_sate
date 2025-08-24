# J2 轨道传播器 - Vulkan 可视化模块
# 此文件配置 Vulkan 可视化系统的构建

cmake_minimum_required(VERSION 3.16)

# 设置项目名称和版本
project(J2OrbitVisualization VERSION 1.0.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 构建选项
option(BUILD_EXAMPLES "Build example programs" ON)

# 设置编译选项
if(MSVC)
    add_compile_options(/W4 /WX- /permissive-)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找必需的包
find_package(Vulkan REQUIRED)
# 尝试通过 CMake Config 包查找 GLFW
find_package(glfw3 CONFIG)

# 如果未找到，则尝试手动查找并创建导入目标
if(NOT glfw3_FOUND)
    message(WARNING "glfw3 config package not found. Trying manual search.")
    set(GLFW_SEARCH_DIRS
        $ENV{VCPKG_ROOT}/installed/x64-windows
        $ENV{VCPKG_ROOT}/installed/x64-windows-static
        $ENV{VCPKG_ROOT}/installed/x86-windows
        "C:/Program Files/glfw"
        "C:/Program Files (x86)/glfw"
    )
    find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h PATH_SUFFIXES include HINTS ${GLFW_SEARCH_DIRS})
    find_library(GLFW_LIBRARY NAMES glfw3 glfw PATH_SUFFIXES lib lib64 HINTS ${GLFW_SEARCH_DIRS})
    if(GLFW_INCLUDE_DIR AND GLFW_LIBRARY)
        add_library(glfw UNKNOWN IMPORTED)
        set_target_properties(glfw PROPERTIES
            IMPORTED_LOCATION "${GLFW_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}"
        )
        set(glfw3_FOUND TRUE)
        message(STATUS "Found GLFW via manual search: ${GLFW_LIBRARY}")
    else()
        message(WARNING "GLFW3 not found via manual search. Falling back to FetchContent to download and build GLFW.")
        include(FetchContent)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            glfw_src
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.3.9
        )
        FetchContent_MakeAvailable(glfw_src)
        # Upstream target is named 'glfw'
        if(NOT TARGET glfw)
            add_library(glfw INTERFACE IMPORTED)
            target_link_libraries(glfw INTERFACE glfw)
        endif()
        set(glfw3_FOUND TRUE)
        message(STATUS "GLFW fetched and added via FetchContent")
    endif()
else()
    # 如果目标名不是 'glfw'，创建别名以保持 target_link_libraries 一致
    if(NOT TARGET glfw)
        if(TARGET glfw::glfw)
            add_library(glfw INTERFACE IMPORTED)
            target_link_libraries(glfw INTERFACE glfw::glfw)
        elseif(TARGET glfw3)
            add_library(glfw INTERFACE IMPORTED)
            target_link_libraries(glfw INTERFACE glfw3)
        endif()
    endif()
endif()
find_package(glm CONFIG QUIET)

if(NOT glm_FOUND)
    message(WARNING "glm config package not found. Falling back to FetchContent.")
    include(FetchContent)
    FetchContent_Declare(
        glm_src
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
    )
    FetchContent_GetProperties(glm_src)
    if(NOT glm_src_POPULATED)
        FetchContent_Populate(glm_src)
    endif()
    # 作为头文件库使用 GLM，避免调用其上游 CMake 配置
    if(NOT TARGET glm::glm)
        add_library(glm INTERFACE)
        target_include_directories(glm INTERFACE ${glm_src_SOURCE_DIR})
        add_library(glm::glm ALIAS glm)
    endif()
    # 标记回退并记录包含目录
    set(GLM_FALLBACK_INCLUDE_DIR ${glm_src_SOURCE_DIR})
    message(STATUS "GLM headers fetched and added as INTERFACE target via FetchContent")
endif()

# 添加 stb_image 库支持
if(NOT TARGET stb_image)
    include(FetchContent)
    FetchContent_Declare(
        stb_image_src
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
    )
    FetchContent_GetProperties(stb_image_src)
    if(NOT stb_image_src_POPULATED)
        FetchContent_Populate(stb_image_src)
    endif()
    # 创建 stb_image 接口库
    add_library(stb_image INTERFACE)
    target_include_directories(stb_image INTERFACE ${stb_image_src_SOURCE_DIR})
    set(STB_IMAGE_INCLUDE_DIR ${stb_image_src_SOURCE_DIR})
    message(STATUS "STB Image headers fetched and added as INTERFACE target via FetchContent")
endif()

# 检查 Vulkan SDK
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "Vulkan SDK not found. Please install Vulkan SDK and set VULKAN_SDK environment variable.")
endif()

# 打印找到的包信息
message(STATUS "Vulkan found: ${Vulkan_LIBRARIES}")
message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "GLFW found (target: glfw)")
message(STATUS "GLM found: ${glm_VERSION}")

# 设置源文件
set(VISUALIZATION_SOURCES
    src/vulkan_renderer.cpp
    src/earth_renderer.cpp
    src/orbit_renderer.cpp
    src/j2_orbit_integration.cpp
    src/visualization_types.cpp
)

# 设置头文件
set(VISUALIZATION_HEADERS
    include/visualization_types.h
    include/vulkan_renderer.h
    include/earth_renderer.h
    include/orbit_renderer.h
    include/j2_orbit_integration.h
)

# 创建可视化静态库
add_library(j2_visualization STATIC
    ${VISUALIZATION_SOURCES}
    ${VISUALIZATION_HEADERS}
)

# 设置目标属性
set_target_properties(j2_visualization PROPERTIES
    OUTPUT_NAME "j2_visualization"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 设置包含目录
target_include_directories(j2_visualization
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>  # 主项目头文件
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../lib/eigen>  # Eigen库
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接库
# 使用生成表达式避免导出时引用到本项目内的 glfw 目标：
# - 构建阶段使用本项目内（或 FetchContent 提供）的 glfw 目标
# - 安装阶段在导出包中引用 glfw::glfw，由消费者通过 find_package(glfw3) 提供
# GLM：若使用回退，则不做 PUBLIC 链接，仅通过包含目录暴露；若非回退，则仍可通过头文件方式使用，无需链接
# 查找主项目的静态库
find_library(J2_ORBIT_PROPAGATOR_LIB
    NAMES j2_orbit_propagator_static
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../build/lib
          ${CMAKE_CURRENT_SOURCE_DIR}/../build/Release
          ${CMAKE_CURRENT_SOURCE_DIR}/../build/Debug
    NO_DEFAULT_PATH
)

if(NOT J2_ORBIT_PROPAGATOR_LIB)
    message(WARNING "j2_orbit_propagator_static library not found. Please build the main project first.")
else()
    message(STATUS "Found j2_orbit_propagator_static: ${J2_ORBIT_PROPAGATOR_LIB}")
endif()

target_link_libraries(j2_visualization
    PUBLIC
        Vulkan::Vulkan
    PRIVATE
        $<BUILD_INTERFACE:glfw>
        $<INSTALL_INTERFACE:glfw::glfw>
        $<BUILD_INTERFACE:stb_image>  # STB Image 库，仅在构建时链接
        ${CMAKE_DL_LIBS}  # 动态链接库支持
        ${J2_ORBIT_PROPAGATOR_LIB}  # 主项目静态库
)
# 如果使用 GLM 回退，仅在构建期加入其包含目录，避免导出时暴露构建路径
if(GLM_FALLBACK_INCLUDE_DIR)
    target_include_directories(j2_visualization PRIVATE ${GLM_FALLBACK_INCLUDE_DIR})
endif()

# 设置编译定义
# SHADER_DIR: 构建时指向构建目录/shaders；安装后指向安装路径下的 shaders
# 通过 BUILD_INTERFACE / INSTALL_INTERFACE 避免将构建路径写入安装导出
target_compile_definitions(j2_visualization
    PRIVATE
        GLM_FORCE_RADIANS
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_ENABLE_EXPERIMENTAL
        J2_BUILD_STATIC  # 告诉 J2 API 使用静态链接
    PUBLIC
        "$<BUILD_INTERFACE:SHADER_DIR=\"${CMAKE_BINARY_DIR}/shaders\">"
        "$<INSTALL_INTERFACE:SHADER_DIR=\"${CMAKE_INSTALL_PREFIX}/share/j2_visualization/shaders\">"
)

# 根据平台设置 Vulkan 平台宏
if(WIN32)
    target_compile_definitions(j2_visualization PRIVATE VK_USE_PLATFORM_WIN32_KHR)
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(j2_visualization PRIVATE VK_USE_PLATFORM_XLIB_KHR)
elseif(APPLE)
    target_compile_definitions(j2_visualization PRIVATE VK_USE_PLATFORM_MACOS_MVK)
endif()

# 添加着色器编译
find_program(GLSLC glslc HINTS ${Vulkan_TOOLS_DIR} PATHS ENV VULKAN_SDK PATH_SUFFIXES bin)

if(NOT GLSLC)
    message(WARNING "glslc not found. Shaders will not be compiled automatically.")
else()
    message(STATUS "Found glslc: ${GLSLC}")
    
    # 设置着色器目录
    set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(SHADER_BINARY_DIR "${CMAKE_BINARY_DIR}/shaders")
    
    # 创建着色器输出目录
    file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})
    
    # 着色器文件列表
    set(SHADER_SOURCES
        ${SHADER_SOURCE_DIR}/earth.vert
        ${SHADER_SOURCE_DIR}/earth.frag
        ${SHADER_SOURCE_DIR}/orbit.vert
        ${SHADER_SOURCE_DIR}/orbit.frag
        ${SHADER_SOURCE_DIR}/satellite.vert
        ${SHADER_SOURCE_DIR}/satellite.frag
    )
    
    # 编译着色器
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME)
        set(SHADER_OUTPUT "${SHADER_BINARY_DIR}/${SHADER_NAME}.spv")
        
        add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${GLSLC} ${SHADER} -o ${SHADER_OUTPUT}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader ${SHADER_NAME}"
        )
        
        list(APPEND SHADER_BINARIES ${SHADER_OUTPUT})
    endforeach()
    
    # 创建着色器目标
    add_custom_target(compile_shaders ALL DEPENDS ${SHADER_BINARIES})
    add_dependencies(j2_visualization compile_shaders)
endif()

# 复制资源文件
set(ASSETS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(ASSETS_BINARY_DIR "${CMAKE_BINARY_DIR}/assets")

if(EXISTS ${ASSETS_SOURCE_DIR})
    file(COPY ${ASSETS_SOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})
endif()

# 构建示例程序（如果启用）
if(BUILD_EXAMPLES)
    add_executable(orbit_visualization_demo
        examples/orbit_visualization_demo.cpp
    )
    
    set_target_properties(orbit_visualization_demo PROPERTIES
        OUTPUT_NAME "orbit_visualization_demo"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    
    target_link_libraries(orbit_visualization_demo
        PRIVATE
            j2_visualization
    )
    
    # 设置工作目录（用于调试）
    set_target_properties(orbit_visualization_demo PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()

# 安装配置
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install prefix" FORCE)
endif()

# 安装库文件
install(TARGETS j2_visualization
    EXPORT J2VisualizationTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件
install(DIRECTORY include/
    DESTINATION include/j2_visualization
    FILES_MATCHING PATTERN "*.h"
)

# 若使用 GLM 回退，将其头文件安装到 include 下，使消费者可用 <glm/...> 直接包含
if(GLM_FALLBACK_INCLUDE_DIR)
    install(DIRECTORY ${GLM_FALLBACK_INCLUDE_DIR}/glm
        DESTINATION include
    )
endif()

# 安装着色器文件
if(EXISTS ${SHADER_BINARY_DIR})
    install(DIRECTORY ${SHADER_BINARY_DIR}/
        DESTINATION share/j2_visualization/shaders
        FILES_MATCHING PATTERN "*.spv"
    )
endif()

# 安装资源文件
if(EXISTS ${ASSETS_BINARY_DIR})
    install(DIRECTORY ${ASSETS_BINARY_DIR}/
        DESTINATION share/j2_visualization/assets
    )
endif()

# 安装示例程序
if(BUILD_EXAMPLES AND TARGET orbit_visualization_demo)
    install(TARGETS orbit_visualization_demo
        RUNTIME DESTINATION bin
    )
endif()

# 导出目标
install(EXPORT J2VisualizationTargets
    FILE J2VisualizationTargets.cmake
    NAMESPACE J2::
    DESTINATION lib/cmake/J2Visualization
)

# 创建配置文件
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/J2VisualizationConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/J2VisualizationConfig.cmake"
    INSTALL_DESTINATION lib/cmake/J2Visualization
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/J2VisualizationConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# 安装配置文件
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/J2VisualizationConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/J2VisualizationConfigVersion.cmake"
    DESTINATION lib/cmake/J2Visualization
)

# 显示构建信息
message(STATUS "========================================")
message(STATUS "J2 Orbit Visualization Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Vulkan SDK: ${VULKAN_SDK}")
message(STATUS "Shader compiler: ${GLSLC}")
message(STATUS "========================================")