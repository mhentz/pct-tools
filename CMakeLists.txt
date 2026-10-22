cmake_minimum_required(VERSION 3.15...3.27)

# scikit-build-core sets these; default them so a plain `cmake -S . -B <dir>` works too
if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME pct_tools)
endif()
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
    set(SKBUILD_PROJECT_VERSION 0.0.0)
endif()

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

option(BUILD_PCT_TOOLS_TESTS "Build C++ tests for pct-tools" OFF)
message(STATUS "Build C++ tests for pct-tools: ${BUILD_PCT_TOOLS_TESTS}")
if(BUILD_PCT_TOOLS_TESTS)
    enable_testing()
endif()

include(FetchContent)

# -------------------------------------------------------------
# Python: only the extension module needs it; the C++ tests do not link Python at all
# -------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Python executable: ${Python_EXECUTABLE} (${Python_VERSION})")

# -------------------------------------------------------------
# OpenMP; on macOS fall back to Homebrew's libomp
# -------------------------------------------------------------
find_package(OpenMP)

if(NOT OpenMP_FOUND AND APPLE)
    execute_process(COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "OpenMP not found; trying Homebrew libomp at ${HOMEBREW_LIBOMP_PREFIX}")

    if(EXISTS ${HOMEBREW_LIBOMP_PREFIX})
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
        find_package(OpenMP REQUIRED)
    else()
        message(FATAL_ERROR "OpenMP not found; install Homebrew's libomp (brew install libomp)")
    endif()
elseif(NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found; install an OpenMP runtime or use a different compiler")
endif()

# -------------------------------------------------------------
# pybind11: provided by scikit-build-core, otherwise fetched
# -------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)

if(SKBUILD)
    find_package(pybind11 REQUIRED)
else()
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0)
    FetchContent_MakeAvailable(pybind11)
endif()

# -------------------------------------------------------------
# Eigen3 and Blosc2 are always fetched at pinned versions
# -------------------------------------------------------------
# Header-only use of Eigen: skip its test, documentation and pkg-config targets.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0)
FetchContent_MakeAvailable(Eigen3)
# Fail the build if any Eigen header under a licence other than MPL-2.0 (e.g. LGPL) is included.
target_compile_definitions(eigen INTERFACE EIGEN_MPL2_ONLY)

# Only the static library is needed; skip Blosc2's own tests, benchmarks and examples.
set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    blosc2
    GIT_REPOSITORY https://github.com/Blosc/c-blosc2.git
    GIT_TAG v2.15.1)
FetchContent_MakeAvailable(blosc2)
include_directories("${blosc2_SOURCE_DIR}/include")

if(BUILD_PCT_TOOLS_TESTS)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1)
    FetchContent_MakeAvailable(catch2)
endif()

add_subdirectory(src/cpp)

if(BUILD_PCT_TOOLS_TESTS)
    add_subdirectory(tests/cpp)
endif()
