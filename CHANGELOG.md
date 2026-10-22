# CHANGELOG

Versions before 1.0.0 were developed in a private repository before the first public
release, and their history is not part of this repository.

<!-- version list -->

## v1.0.0 (2026-10-22)

### Breaking

* feat: add the pct-tools library

Proton CT reconstruction tools: exact chord-length system matrix, most likely
path, Lipschitz step estimate, gradient-descent reconstruction and compressed
sparse-matrix I/O, with C++ (Catch2) and Python (pytest) test suites.

BREAKING CHANGE: array arguments must already have dtype float32 (int32 for
sparse indices); other dtypes and plain lists raise TypeError instead of being
converted silently. ([`ca0a537`](https://github.com/mhentz/pct-tools/commit/ca0a537930ed014733d489bced998cc46caac88c))

### Continuous Integration

* ci: build wheels in manylinux_2_28 images

numpy and scipy no longer publish manylinux2014 wheels, so the wheel test step
in the default image fell back to building numpy from source and failed on its
compiler requirement. ([`fa59682`](https://github.com/mhentz/pct-tools/commit/fa5968283781b72f7698e39104ebe18dfe3b5126))
