# 集成测试
file(GLOB INTEGRATION_TEST_SOURCES "*.cpp")

if(INTEGRATION_TEST_SOURCES)
  add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})
  target_include_directories(integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/eigen})
  target_link_libraries(integration_tests PRIVATE GTest::gtest GTest::gtest_main)
  add_test(NAME integration_tests COMMAND integration_tests)
endif()