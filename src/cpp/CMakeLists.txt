file(GLOB_RECURSE SOURCES "src/*.cpp")

set(PCT_TOOLS_LIBRARIES Eigen3::Eigen blosc2_static OpenMP::OpenMP_CXX)
if(UNIX AND NOT APPLE)
    list(APPEND PCT_TOOLS_LIBRARIES dl)
endif()

# Core library: the numerical code without any Python dependency. The extension module
# adds the pybind11 bindings on top; the C++ tests (tests/cpp) link it directly.
add_library(_core OBJECT ${SOURCES})
target_include_directories(_core PUBLIC include include/matrix)
target_link_libraries(_core PUBLIC ${PCT_TOOLS_LIBRARIES})
set_target_properties(_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(_core PRIVATE -Wall -Wextra)

pybind11_add_module(_ext MODULE bindings.cpp WITH_SOABI)
target_link_libraries(_ext PRIVATE _core)
target_compile_definitions(_ext PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_options(_ext PRIVATE -Wall -Wextra)

install(TARGETS _ext LIBRARY DESTINATION pct_tools)
