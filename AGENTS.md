# trueform

Real-time geometric processing. Header-only C++17, parallel via oneTBB. Python bindings (nanobind), TypeScript/WASM bindings (Emscripten).

Arrangements, booleans, registration, remeshing, spatial queries — at interactive speed on million-polygon meshes. Exact and robust via int32 predicates with SoS.

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `include/trueform/` | Header-only C++17 core (modules: core, topology, spatial, geometry, remesh, intersect, cut, exact, clean, reindex, io, random) |
| `python/` | Python bindings (nanobind), wrappers, tests |
| `typescript/` | TypeScript/WASM bindings (Emscripten embind), wrappers, tests |
| `tests/` | C++ Catch2 test suite |
| `benchmarks/` | Performance comparisons |
| `docs/` | Documentation (Nuxt Content, 3 language tracks: cpp, ts, py) |
| `agents/` | AI agent knowledge base (patterns, conventions, API reference) |
| `nuget/` | NuGet package configuration |

## Build Commands

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
pytest python/tests
```

### TypeScript
```bash
# Requires Emscripten SDK
emcmake cmake -B build-wasm typescript/
cmake --build build-wasm --parallel
```

### Benchmarks
```bash
cmake -B build -DTF_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target benchmarks
```

## C++ Code Style

- **Parallel by default**: `tf::parallel_for_each`, `tf::parallel_copy`, `tf::parallel_fill` — never raw loops for bulk ops
- **Range composition**: Lazy views (`make_indirect_range`, `make_mapped_range`, `make_offset_block_range`) — no intermediate allocations
- **View/buffer split**: Buffers own memory, views reference it. Functions accept views.
- **Policy composition**: `polygons | tf::tag(tree) | tf::tag(fm)` — attach precomputed structures via pipe
- **`build()` pattern**: Stateful structures default-construct empty, populate via `build(...)`, query via accessors
- **Sentinel maps**: `buffer[n]` filled with sentinel for O(1) integer-keyed lookup — never `std::unordered_map` in hot paths
- **Naming**: `snake_case` functions and types, trailing return types, no type aliases in library code
- **Templates**: `typename Index = tf::none_t` + `if constexpr` for zero-cost type deduction
- **Format**: `.clang-format` BasedOnStyle: LLVM

## Adding a Feature (Abbreviated)

1. C++ header in `include/trueform/<module>/`
2. Add to umbrella header with `// IWYU pragma: export`
3. Catch2 test in `tests/<module>/`
4. Python: nanobind binding (one `.cpp` per type combo) + Python wrapper + pytest test
5. TypeScript: embind binding (sync + async) + TS wrapper (sync + async) + test
6. Docs: update `docs/content/{cpp,ts,py}/2.modules/`

See `agents/feature_lifecycle.md` for the full checklist with naming transformations.

## Deep Knowledge

The `agents/` directory contains detailed knowledge documents:

| Document | Topic |
|----------|-------|
| `usage_cpp.md` | C++ usage patterns with code examples |
| `usage_typescript.md` | TypeScript usage patterns |
| `usage_python.md` | Python usage patterns |
| `cpp_core_architecture.md` | Type system, memory model, ranges, policies |
| `cpp_engineering_philosophy.md` | Engineering principles, parallel patterns, style rules |
| `cpp_modules.md` | Per-module API reference |
| `typescript_layer.md` | WASM bindings, thread safety, async patterns |
| `python_layer.md` | Nanobind bindings, dtype dispatch |
| `feature_lifecycle.md` | End-to-end feature addition checklist |
| `documentation_architecture.md` | How docs are structured |
