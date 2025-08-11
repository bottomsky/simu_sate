# 单元测试
file(GLOB UNIT_TEST_SOURCES "*.cpp")

if(UNIT_TEST_SOURCES)
  add_executable(unit_tests ${UNIT_TEST_SOURCES})
  target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/eigen)
  target_link_libraries(unit_tests PRIVATE GTest::gtest GTest::gtest_main)
  add_test(NAME unit_tests COMMAND unit_tests)
endif()