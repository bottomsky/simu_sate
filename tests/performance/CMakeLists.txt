# 性能测试
file(GLOB PERFORMANCE_TEST_SOURCES "*.cpp")

# 包含星座传播器相关源文件
set(CONSTELLATION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/constellation_propagator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/j2_orbit_propagator.cpp
)

# 检查CUDA编译器是否可用
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/j2_cuda_kernels.cu)
    message(STATUS "CUDA Toolkit found, enabling GPU acceleration for performance tests")
else()
    set(CUDA_SOURCES)
    message(STATUS "CUDA Toolkit not found - performance tests will run on CPU only")
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

    # CUDA特定设置（仅在检测到CUDA编译器时）
    if(CUDAToolkit_FOUND)
        set_target_properties(performance_tests PROPERTIES
            CUDA_RUNTIME_LIBRARY Shared
        )
    endif()
    
    # 链接库
    target_link_libraries(performance_tests PRIVATE 
        GTest::gtest 
        GTest::gtest_main
        Eigen3::Eigen
    )
    
    add_test(NAME performance_tests COMMAND performance_tests)
endif()