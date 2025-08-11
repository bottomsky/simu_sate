# 性能测试
file(GLOB PERFORMANCE_TEST_SOURCES "*.cpp")

if(PERFORMANCE_TEST_SOURCES)
  add_executable(performance_tests ${PERFORMANCE_TEST_SOURCES})
  target_include_directories(performance_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
  # 性能测试通常不需要断言主程序退出码，因此也可以不依赖gtest。
  # 这里仍然链接gtest以便统一框架下进行性能基准记录。
  target_link_libraries(performance_tests PRIVATE GTest::gtest GTest::gtest_main)
  add_test(NAME performance_tests COMMAND performance_tests)
endif()