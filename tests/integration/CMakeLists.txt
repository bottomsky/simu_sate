# 集成测试
file(GLOB INTEGRATION_TEST_SOURCES "*.cpp")

if(INTEGRATION_TEST_SOURCES)
  add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})
  target_include_directories(integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/eigen)
  target_link_libraries(integration_tests PRIVATE GTest::gtest GTest::gtest_main)
  
  # 链接被测源码
  target_sources(integration_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/j2_orbit_propagator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/constellation_propagator.cpp
  )

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