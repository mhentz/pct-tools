# Contributing

Bug reports, questions and pull requests are welcome. For anything larger than a small fix, please open an issue first so the change can be discussed before you spend time on it.

## Setting up

The README's [Local development](README.md#local-development) section describes how to build the C++ code with plain CMake and how to make an editable install of the Python package. The `Makefile` wraps the common commands:

```sh
make cpp-test   # configure, build and run the Catch2 suite in build-cpp/
make py-build   # install the package with its test extras into .venv
make py-test    # run the pytest suite against the installed package
```

Formatting and linting are handled by [pre-commit](https://pre-commit.com/) with ruff for Python and clang-format for C++. Install the hooks once with

```sh
uvx pre-commit install
```

and they run on every commit; `uvx pre-commit run -a` checks the whole tree. CI runs the same tools and fails on any difference.

## Commit messages

Releases are cut automatically by [python-semantic-release](https://python-semantic-release.readthedocs.io/), which reads the commit messages on `main` to decide whether a release is due and what the new version is. Commit messages therefore follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <short summary>

<optional body explaining what and why>
```

The type decides the version bump:

| Type | Effect on the next version |
| --- | --- |
| `fix:` | patch release (1.0.0 → 1.0.1) |
| `feat:` | minor release (1.0.0 → 1.1.0) |
| any type with a `BREAKING CHANGE:` footer | major release (1.0.0 → 2.0.0) |
| `docs:`, `ci:`, `test:`, `refactor:`, `chore:`, `build:`, `perf:`, `style:` | no release |

Keep the summary in the imperative and under about 70 characters. The body is optional but appreciated for anything non-obvious; it ends up in the changelog verbatim.

## Pull requests

Open pull requests against `main`. The CI workflow runs on every pull request: linting, the C++ and Python test suites on Linux and macOS, and a build of the API reference with warnings treated as errors. A pull request that changes only the documentation and Markdown files skips the lint and test jobs. Wheels are not built for pull requests. The first run from a new contributor needs to be approved by a maintainer before it starts.

Nothing is published from a pull request. When a pull request is merged into `main`, the workflow runs again on the merge; if the commits warrant a release it commits the updated `CHANGELOG.md` and version number, tags the new version and uploads the wheels to PyPI.

Please keep pull requests focused on one change, add or adjust tests where behaviour changes, and update the README if the user-facing interface changes.
