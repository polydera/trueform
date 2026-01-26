# Contributing to trueform

We welcome contributions—bug fixes, new features, documentation, examples, and benchmarks.

## Start with GitHub Issues

Browse [open issues](https://github.com/polydera/trueform/issues) labeled by difficulty:

- **🟢 `difficulty: easy`** — Good first issues. Bug fixes, documentation, straightforward enhancements.
- **🟡 `difficulty: medium`** — Features, refactors, complex bugs. Requires familiarity with the codebase.
- **🔴 `difficulty: hard`** — Novel algorithms, architectural changes, deep geometric reasoning.

Pick an issue, comment that you're working on it, and get started.

**Interested in joining our team?** If you successfully tackle a hard-difficulty issue, we'd love to talk. Reach out to [info@polydera.com](mailto:info@polydera.com).

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `include/` | Header-only C++17 core library |
| `python/` | Python bindings, tests, and examples |
| `vtk/` | VTK integration library and examples |
| `examples/` | C++ examples |
| `benchmarks/` | Performance comparisons |
| `tests/` | C++ tests |
| `docs/` | Documentation |
| `verify/` | Build and test verification scripts |

## Building

### Core Library

Header-only. No build required—just include and link `tf::trueform`.

### C++ Tests

```bash
cmake -B build -DTF_BUILD_TESTS=ON
cmake --build build --parallel --target trueform_tests
ctest --test-dir build --output-on-failure
```

### Python

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 pip wheel . -w dist
pip install dist/trueform-*.whl
```

Run tests:
```bash
pytest python/tests
```

### VTK Integration

```bash
cmake -B build -DTF_BUILD_VTK_INTEGRATION=ON
cmake --build build --parallel
```

VTK examples:
```bash
cmake -B build -DTF_BUILD_VTK_INTEGRATION=ON -DTF_BUILD_VTK_EXAMPLES=ON
cmake --build build --parallel --target trueform_vtk_examples
```

### C++ Examples

```bash
cmake -B build -DTF_BUILD_EXAMPLES=ON
cmake --build build --parallel --target trueform_examples
```

### Benchmarks

```bash
cmake -B build -DTF_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target benchmarks
```

## Workflow

**1. Fork and branch**
```bash
git checkout -b feature/your-feature-name
```

**2. Make your changes** following the code guidelines below.

**3. Commit with clear messages**

One-liner summary, then explanation if needed:
```bash
git commit -m "Add linear terms to curvature quadratic fitting

Extends the fitted surface from ax² + bxy + cy² to include dx + ey.
Linear terms absorb local plane tilt, improving robustness when the
neighborhood center doesn't coincide with the vertex."
```

**4. Verify build and tests pass**

```bash
python -m verify
```

This clones the repo to a temp directory and verifies everything works from scratch: C++ build and installation, `find_package` integration, Python pip install, and the full test suite.

Options:
- `--skip-vtk` — Skip VTK integration if you don't have VTK installed
- `--skip-python` — Skip Python package verification
- `--keep` — Keep build artifacts for debugging
- `--work-dir <path>` — Use a specific working directory
- `--toolchain-file <path>` — CMake toolchain file (also reads `CMAKE_TOOLCHAIN_FILE` env var)

**5. Push and open a pull request**
```bash
git push origin feature/your-feature-name
```

Reference the issue (e.g., "Fixes #42") and describe what you did.

## Code Guidelines

Trueform is built on composability and zero-copy semantics. When contributing:

- **Build on existing algorithms.** Implement new functionality using already-written parallel algorithms. Don't reinvent parallelism—compose it.
- **One function, one file.** Each user-facing function in `tf::` namespace gets its own file in the module directory. Include only the functions you use, not entire modules.
- **Small functions, delegated responsibilities.** Break complex operations into small, focused functions. Each should do one thing clearly.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for template arguments.
- **Minimize comments.** Code should be self-explanatory. Comment only non-obvious algorithmic choices.

## Documentation

**Docstrings:** Add docstrings to new functions:
- C++: Doxygen-style (`/// @brief`, `/// @param`, `/// @return`) in header files
- Python: Docstrings with parameter descriptions and return types

**Docs site:** If you add a public API, update `docs/content/` with:
- What the function does (brief, one sentence)
- Example usage
- Return value description if non-obvious

Follow the existing structure in the modules documentation.

## Licensing

By contributing, you certify that:

1. Your work is provided under the **PolyForm Noncommercial License 1.0.0** for noncommercial use.
2. XLAB may offer your work under its **commercial license** without requiring additional paperwork from you.

If your employer owns your work, ensure they're comfortable with this arrangement.

## Questions?

- **Technical questions:** [GitHub Discussions](https://github.com/polydera/trueform/discussions)
- **Bug reports:** [Open an issue](https://github.com/polydera/trueform/issues)
- **Commercial licensing or jobs:** [info@polydera.com](mailto:info@polydera.com)

---

Thank you for contributing to trueform.
