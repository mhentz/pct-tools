# Shortcuts for the commands in the README's "Running the tests" and "Local development"
# sections. The C++ targets use the plain CMake build in build-cpp/, without Python; the
# Python targets install the package into the project's .venv with uv.

CPP_BUILD_DIR := build-cpp
JOBS ?= 4

.PHONY: cpp-build cpp-test py-build py-test

# Configure once; later builds reuse the cache and are incremental.
$(CPP_BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(CPP_BUILD_DIR) -DBUILD_PCT_TOOLS_TESTS=ON -DCMAKE_BUILD_TYPE=Release

cpp-build: $(CPP_BUILD_DIR)/CMakeCache.txt
	cmake --build $(CPP_BUILD_DIR) -j $(JOBS) --target pct_tools_tests

cpp-test: cpp-build
	ctest --test-dir $(CPP_BUILD_DIR) -R pct_tools_tests --output-on-failure

py-build:
	uv pip install '.[test]'

# Runs against the package currently installed in .venv; run py-build first after
# changing the sources.
py-test:
	.venv/bin/pytest tests/python -v
