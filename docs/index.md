# pct-tools

Tools for proton CT image reconstruction with very large, sparse system matrices.

The package is a thin Python layer over a C++ extension. The system matrix of a proton CT scan has one row per proton and one column per voxel, and for realistic scans it does not fit in memory. `pct-tools` therefore treats it as a set of row blocks, one per block of proton trajectories. Each block is stored compressed on disk and is read back only while it is needed; the operations loop over blocks in parallel with OpenMP.

The trajectories are typically produced with [protonct-sim](https://github.com/mhentz/protonct-sim), a Geant4 application that simulates proton CT acquisitions.

## Installation

```sh
uv pip install pct-tools
```

Wheels are published for CPython 3.12 on Linux (x86_64, aarch64) and macOS (arm64). Building from source, and the development set-up, are described in the [README](https://github.com/mhentz/pct-tools#installation).

## Usage

```{include} ../README.md
:start-after: "## Usage"
:end-before: "## Running the tests"
```

## API reference

Lengths are in millimetres and angles in radians throughout. Array arguments must be numpy arrays of dtype `float32` (`int32` for sparse indices); other dtypes raise `TypeError` rather than being converted. Submodules are not imported by the package itself; import the one you need.

| Module | Purpose |
| --- | --- |
| {mod}`pct_tools.matrix` | Chord lengths of trajectories through a voxel grid, i.e. the system-matrix entries; builds and stores a block. |
| {mod}`pct_tools.mlp` | Most likely path of a proton from its entry and exit position and direction. |
| {mod}`pct_tools.io` | Binary matrix and vector formats, compressed with Blosc2. |
| {mod}`pct_tools.recon` | One steepest-descent step and the squared residual norm over all blocks. |
| {mod}`pct_tools.lipschitz` | `A.T @ A @ x` over all blocks, for estimating the step size with the power method. |
| {mod}`pct_tools.utils` | Recompression and row subsampling of stored blocks. |

```{toctree}
:maxdepth: 1
:hidden:
:caption: API reference

api/matrix
api/mlp
api/io
api/recon
api/lipschitz
api/utils
```

```{toctree}
:hidden:
:caption: Project

Repository <https://github.com/mhentz/pct-tools>
PyPI <https://pypi.org/project/pct-tools/>
Changelog <https://github.com/mhentz/pct-tools/blob/main/CHANGELOG.md>
```
