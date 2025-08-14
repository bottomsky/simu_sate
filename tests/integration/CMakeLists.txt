# 集成测试
file(GLOB INTEGRATION_TEST_SOURCES "*.cpp")

# 检查CUDA是否可用（由父项目启用）
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST languages)
  set(CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/j2_cuda_kernels.cu)
  message(STATUS "CUDA language enabled, including GPU acceleration for integration tests")
else()
  set(CUDA_SOURCES)
  message(STATUS "CUDA language not enabled - integration tests will run on CPU only")
endif()

if(INTEGRATION_TEST_SOURCES)
  add_executable(integration_tests 
    ${INTEGRATION_TEST_SOURCES}
    ${CUDA_SOURCES}
  )
  target_include_directories(integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/eigen)
  target_link_libraries(integration_tests PRIVATE GTest::gtest GTest::gtest_main)
  
  # 链接被测源码
  target_sources(integration_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/j2_orbit_propagator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/constellation_propagator.cpp
  )

  # CUDA特定设置（仅在CUDA可用时）
  if("CUDA" IN_LIST languages)
    set_target_properties(integration_tests PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared
      CUDA_SEPARABLE_COMPILATION ON
    )
    # 为CUDA编译器添加适当的选项
    target_compile_options(integration_tests PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    # 为C++编译单元提供CUDA运行时头与库
    target_compile_definitions(integration_tests PRIVATE HAVE_CUDA_TOOLKIT=1)
    target_link_libraries(integration_tests PRIVATE CUDA::cudart)
  endif()

  add_test(NAME integration_tests COMMAND integration_tests)

  # 生成一个额外可执行文件用于参数扫描
  add_executable(integration_param_sweep parameter_sweep_test.cpp)
  target_include_directories(integration_param_sweep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/eigen)
  target_link_libraries(integration_param_sweep PRIVATE GTest::gtest GTest::gtest_main)
  target_sources(integration_param_sweep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/j2_orbit_propagator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/constellation_propagator.cpp
  )
  add_test(NAME integration_param_sweep COMMAND integration_param_sweep)
endif()