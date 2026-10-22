# Native test suite (Catch2). Links the core library only: no Python, no pybind11.
file(GLOB TEST_SOURCES "*.cpp")

add_executable(pct_tools_tests ${TEST_SOURCES})
target_link_libraries(pct_tools_tests PRIVATE _core Catch2WithMain)
target_compile_options(pct_tools_tests PRIVATE -Wall -Wextra)

add_test(NAME pct_tools_tests COMMAND pct_tools_tests)
