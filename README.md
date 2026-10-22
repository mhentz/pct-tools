# pct-tools — Tools for proton CT image reconstruction

[![PyPI package](https://img.shields.io/pypi/v/pct-tools.svg)](https://pypi.org/project/pct-tools/)

This Python package contains tools used in proton CT image reconstructions. They may come in handy when dealing with very large and sparse system matrices.

`pct-tools` is a Python wrapper around the C++ extension `_ext` that includes [pybind11](https://pybind11.readthedocs.io/en/stable) bindings and is built using [scikit-build-core](https://github.com/scikit-build/scikit-build-core).

## Overview

The system matrix of a proton CT scan has one row per proton and one column per voxel, and for realistic scans it does not fit in memory. `pct-tools` therefore treats it as a set of row blocks, one per block of proton trajectories. Each block is stored compressed on disk and is read back only while it is needed; the operations below loop over blocks in parallel with OpenMP.

| Module                | Purpose                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `pct_tools.matrix`    | Chord lengths of trajectories through a voxel grid, i.e. the system-matrix entries; builds and stores a block. |
| `pct_tools.mlp`       | Most likely path of a proton from its entry and exit position and direction.                     |
| `pct_tools.io`        | Binary matrix and vector formats, compressed with Blosc2 ([layout](src/pct_tools/io.py)).       |
| `pct_tools.recon`     | One steepest-descent step and the squared residual norm over all blocks.                         |
| `pct_tools.lipschitz` | `A.T @ A @ x` over all blocks, for estimating the step size with the power method.               |
| `pct_tools.utils`     | Recompression and row subsampling of stored blocks.                                              |

Lengths are in millimetres and angles in radians throughout. The voxel indexing and sampling conventions are documented in the module docstring of `pct_tools.matrix`.

## Installation

The commands below use [uv](https://docs.astral.sh/uv/), which is typically much faster than pip at resolving and installing packages. Dropping the `uv` prefix should work just as well with plain pip.

Unlike pip, `uv pip install` expects a virtual environment and will not install into the system Python unless given `--system`. To create one with a specific Python version (3.12 or later is required), run:

```sh
uv venv --python 3.12
source .venv/bin/activate
```

uv downloads the requested Python version if it isn't already installed. It also finds a `.venv` in the current or a parent directory without activation, though activating it makes `python` and `pytest` point to the environment.

The package is pre-built for several platforms and available via [PyPI](https://pypi.org/project/pct-tools):

```sh
uv pip install pct-tools
```

If the package isn't available for your platform, compile it locally after cloning the repository:

```sh
git clone https://github.com/mhentz/pct-tools.git
cd pct-tools
uv pip install .
```

Building from source needs CMake, a C++17 compiler and an OpenMP runtime. On macOS install the latter with `brew install libomp`. Eigen, Blosc2 and pybind11 are downloaded by CMake at pinned versions during the build.

## Usage

The example below builds the system-matrix block for two protons crossing a small image, stores it together with the measured water-equivalent path lengths, and takes one gradient step towards the reconstructed image. A real reconstruction repeats the last two calls over many blocks and iterations.

```python
import numpy as np

from pct_tools.io import read_compressed_matrix, store_vector
from pct_tools.matrix import compute_and_store_sparse_matrix
from pct_tools.recon import gradient_descent, squared_residuals

# A 2 x 3 x 3 voxel image of 1 mm voxels: 2 slices along y, 3 pixels along x and z.
# The corner of voxel (0, 0, 0) sits at x = 3, y = 0, z = 0; x decreases with the j index.
img_shape = (2, 3, 3)
origin = (3.0, 0.0, 0.0)

# Two straight trajectories along z, sampled every 0.25 mm.
z = np.arange(0.0, 3.01, 0.25, dtype=np.float32)
x = np.stack([np.full_like(z, 1.5), np.full_like(z, 2.5)])
y = np.stack([np.full_like(z, 0.5), np.full_like(z, 1.5)])
z = np.stack([z, z])

compute_and_store_sparse_matrix(
    "block_000.mat", x, y, z, img_shape, origin, pixel_width=1.0, slice_thickness=1.0
)
A = read_compressed_matrix("block_000.mat")  # scipy CSR matrix of shape (2, 18)

# Measured water-equivalent path lengths, one per proton.
store_vector(np.array([3.0, 3.0], dtype=np.float32), "block_000.vec")

x_image = np.zeros(A.shape[1], dtype=np.float32)
x_image = gradient_descent(x_image, 0.1, ["block_000.mat"], ["block_000.vec"])
print(squared_residuals(x_image, ["block_000.mat"], ["block_000.vec"]))
```

Every function that loops over blocks or samples takes an `n_threads` keyword. The optional `verbosity_level` arguments print progress from the C++ side; 0 is silent.

Array arguments must be numpy arrays of dtype `float32` (`int32` for the sparse indices passed to `construct_matrix`). Other dtypes and plain lists raise `TypeError` rather than being converted: the C++ side works in `float32`, and a silent conversion would copy the array, which for a block of trajectories or an image vector can be hundreds of megabytes. Convert once with `astype(np.float32)` where the data is produced.

The chord lengths are exact for the sampled polyline as long as no segment between consecutive samples crosses more than one voxel boundary along any axis; a step of half the smallest voxel dimension is a safe choice, and segments that cross more raise `ValueError`. `examples/benchmark_chords.py` times the system-matrix construction for a block of trajectories at several steps and thread counts.

## Running the tests

There are two suites with distinct responsibilities:

- `tests/cpp`: the numerical code, tested in C++ with [Catch2](https://github.com/catchorg/Catch2) against independent reference implementations (chord lengths, most likely path, thread-count consistency). It links the core library only, without Python, and runs through the [C++-only build](#c-only-build) described below.
- `tests/python`: the Python package, tested with pytest. It covers what the bindings and wrappers add on top of the C++ code: argument conversion, result shapes and dtypes, I/O round trips, exception propagation and the agreement between the different entry points. It does not repeat the numerical checks and is expected to pass.

```sh
uv pip install .[test]
pytest tests/python
```

## Local development

### C++-only build

The C\++ code can be configured, built and tested with plain CMake, without Python packaging. This is the quickest way to iterate on the C++ sources and to see compiler warnings (`-Wall -Wextra` are enabled):

```sh
cmake -S . -B build-cpp -DBUILD_PCT_TOOLS_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j 4 --target pct_tools_tests
ctest --test-dir build-cpp -R pct_tools_tests --output-on-failure
```

The first configure downloads Eigen, Blosc2, pybind11 and Catch2 and takes about a minute; subsequent builds are incremental. Dropping `--target pct_tools_tests` also builds the `_ext` extension module, the same code the wheel ships. To see only warnings from this project's sources rather than from the dependencies:

```sh
cmake --build build-cpp -j 4 2>&1 | grep -A3 'src/cpp/.*warning:'
```

A Debug configuration (`-DCMAKE_BUILD_TYPE=Debug`) or a separate build directory with `-DCMAKE_CXX_FLAGS=-fsanitize=address` works the same way.

### Editor support

For an editable install, run `uv pip install -e .`. The build writes a `compile_commands.json` to `build/<wheel tag>/`, which editors using clangd or similar tooling may need in order to resolve the C++ include paths. A symlink at the repository root, which is gitignored, is usually enough:

```sh
ln -s build/cp312-cp312-macosx_27_0_arm64/compile_commands.json .
```

The directory name depends on your Python version and platform, so adjust it accordingly.

## Changelog

[CHANGELOG.md](CHANGELOG.md) is generated from the commit history by [python-semantic-release](https://python-semantic-release.readthedocs.io/) and is not edited by hand.

## Third-party software

The extension module links [Eigen](https://gitlab.com/libeigen/eigen), [c-blosc2](https://github.com/Blosc/c-blosc2) and [pybind11](https://github.com/pybind/pybind11), and the wheels bundle an OpenMP runtime (libomp on macOS, libgomp on Linux). Their licences and copyright notices are reproduced in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Licence

pct-tools is released under the [MIT licence](LICENSE).
