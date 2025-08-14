# 性能测试
file(GLOB PERFORMANCE_TEST_SOURCES "*.cpp")

# 包含星座传播器相关源文件
set(CONSTELLATION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/constellation_propagator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/j2_orbit_propagator.cpp
)

# 检查CUDA是否可用（由父项目启用）
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST languages)
    set(CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/j2_cuda_kernels.cu)
    message(STATUS "CUDA language enabled, including GPU acceleration for performance tests")
else()
    set(CUDA_SOURCES)
    message(STATUS "CUDA language not enabled - performance tests will run on CPU only")
endif()

if(PERFORMANCE_TEST_SOURCES)
    add_executable(performance_tests 
        ${PERFORMANCE_TEST_SOURCES} 
        ${CONSTELLATION_SOURCES}
        ${CUDA_SOURCES}
    )
    
    target_include_directories(performance_tests PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/eigen
    )
    
    # 设置编译标志
    target_compile_features(performance_tests PRIVATE cxx_std_17)
    
    # 启用SIMD优化（仅限C++源文件）
    if(MSVC)
        target_compile_options(performance_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    else()
        target_compile_options(performance_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2> $<$<COMPILE_LANGUAGE:CXX>:-mfma>)
    endif()

    # CUDA特定设置（仅在CUDA可用时）
    if("CUDA" IN_LIST languages)
        set_target_properties(performance_tests PROPERTIES
            CUDA_RUNTIME_LIBRARY Shared
            CUDA_SEPARABLE_COMPILATION ON
        )
        # 为CUDA编译器添加适当的选项
        target_compile_options(performance_tests PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
        # 为C++编译单元提供CUDA运行时头与库
        target_compile_definitions(performance_tests PRIVATE HAVE_CUDA_TOOLKIT=1)
        target_link_libraries(performance_tests PRIVATE CUDA::cudart)
    else()
        # 当CUDA被禁用时，确保不定义CUDA宏
        target_compile_definitions(performance_tests PRIVATE HAVE_CUDA_TOOLKIT=0)
    endif()
    
    # 链接库
    target_link_libraries(performance_tests PRIVATE 
        GTest::gtest 
        GTest::gtest_main
    )
    # Eigen 为头文件库，某些环境不会提供 Eigen3::Eigen 目标，存在时再链接
    if(TARGET Eigen3::Eigen)
        target_link_libraries(performance_tests PRIVATE Eigen3::Eigen)
    endif()
    
    add_test(NAME performance_tests COMMAND performance_tests)
endif()