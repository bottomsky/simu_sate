# 单元测试
include(FetchContent)

# 下载 nlohmann/json 单个头文件版本
set(JSON_HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/json_include)
file(MAKE_DIRECTORY ${JSON_HEADER_DIR})

if(NOT EXISTS ${JSON_HEADER_DIR}/nlohmann/json.hpp)
  file(MAKE_DIRECTORY ${JSON_HEADER_DIR}/nlohmann)
  file(DOWNLOAD 
    https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
    ${JSON_HEADER_DIR}/nlohmann/json.hpp
    SHOW_PROGRESS
  )
endif()

file(GLOB UNIT_TEST_SOURCES "*.cpp")

if(UNIT_TEST_SOURCES)
  add_executable(unit_tests ${UNIT_TEST_SOURCES})
  target_include_directories(unit_tests PRIVATE ${JSON_HEADER_DIR})
  target_link_libraries(unit_tests PRIVATE
    j2_orbit_propagator_static
    GTest::gtest
    GTest::gtest_main
  )
  add_test(NAME unit_tests COMMAND unit_tests)
endif()
