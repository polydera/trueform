# Feature Lifecycle: Adding a Feature Across All Languages

Traced from two real features: `make_sharp_edges` and `chamfer_error`. This documents the end-to-end path and common gaps.

---

## The Full Pipeline

| Step | Location | Naming | Notes |
|------|----------|--------|-------|
| 1. C++ Header | `include/trueform/<module>/<function>.hpp` | `snake_case` | Template, policy-based |
| 2. C++ Umbrella | `include/trueform/<module>.hpp` | `// IWYU pragma: export` | Add include line |
| 3. C++ Test | `tests/<module>/test_<function>.cpp` | Catch2 | `TEMPLATE_TEST_CASE` for type combos |
| 4. C++ Docs | `docs/content/cpp/2.modules/<NN>.<module>.md` | C++ examples | Signature + example |
| 5. Python Binding Header | `python/include/trueform/python/<module>/<function>.hpp` | Template wrapper | Handles transformation dispatch |
| 6. Python Binding C++ | `python/src/<module>/<function>_<types>.cpp` | `{func}_{idx}{ngon}{real}{dims}d` | One file per type combo |
| 7. Python Registration | `python/src/<module>.cpp` | `register_<function>(module)` | Add to submodule |
| 8. Python Wrapper | `python/src/trueform/_<module>/<function>.py` | `snake_case` | Dispatch via `build_suffix()` |
| 9. Python `__init__.py` | `python/src/trueform/__init__.py` | Public export | `from ._<module> import ...` |
| 10. Python Test | `python/tests/test_<function>.py` | pytest | `@parametrize` for type combos |
| 11. Python Docs | `docs/content/py/2.modules/<NN>.<module>.md` | Python examples | Match C++ docs structure |
| 12. TS Binding C++ | `typescript/cpp/src/<module>/<function>.cpp` | `EMSCRIPTEN_BINDINGS` | sync + async (dispatch_*) |
| 13. TS Wrapper Sync | `typescript/src/<module>/sync.ts` | `camelCase` | Calls `native().<function>(...)` |
| 14. TS Wrapper Async | `typescript/src/<module>/async.ts` | `camelCase` | `dispatcher().run(...)` |
| 15. TS Exports | `typescript/src/manual.ts` + `typescript/src/async/index.ts` | Public export | `manual.ts` is the export surface (`index.ts` only re-exports it + auto-init); async twins export from `async/index.ts` |
| 16. TS Test | `typescript/tests/test_<module>.mjs` | assert-based | Manual lifecycle (`.delete()`); register the file in `tests/run.mjs` |
| 17. TS Docs | `docs/content/ts/2.modules/<NN>.<module>.md` | TS examples | Match C++ docs structure |

---

## Key Transformations

### Naming
- C++: `make_sharp_edges` → Python: `sharp_edges` → TS: `sharpEdges`
- C++: `chamfer_error` → Python: `chamfer_error` → TS: `chamferError`
- Pattern: drop `make_` prefix, `snake_case` for Python, `camelCase` for TS

### Type Parameters
- C++ `template <typename Index, typename RealT>` → Python `_intfloat3d` suffix → TS single binding (float only)
- C++ `rad<T>` / `deg<T>` → Python `float` (radians) → TS `number` (degrees)
- C++ `polygons_buffer<...>` → Python `(np.ndarray, np.ndarray)` tuple → TS `Mesh` object

### Return Types
- C++ `blocked_buffer<Index, 2>` → Python `np.ndarray` shape `[N, 2]` → TS `NDArrayInt32`
- C++ `std::tuple<mesh, labels, face_labels>` → Python tuple of numpy arrays → TS `LabeledCutResult` interface

### Python Dispatch Pattern
```python
meta = extract_meta(mesh)
suffix = build_suffix(meta)  # e.g., "int3float3d"
func = getattr(_trueform.module, f"{operation}_{suffix}")
result = func(mesh._wrapper, ...)
```

### TS Sync/Async Pattern
```typescript
// Sync
export function sharpEdges(m: Mesh, angleDeg: number): NDArrayInt32 {
    return new NDArray(native().sharp_edges(m._handle, angleDeg), "int32");
}

// Async
export async function sharpEdges(m: Mesh, angleDeg: number): Promise<NDArrayInt32> {
    return dispatcher().run(
        () => native().dispatch_sharp_edges(m._handle, angleDeg),
        (raw) => new NDArray(raw, "int32"));
}
```

---

## Coverage Gaps Found

| Feature | C++ Header | C++ Test | Py Binding | Py Test | Py Docs | TS Binding | TS Test | TS Docs |
|---------|-----------|----------|------------|---------|---------|------------|---------|---------|
| `make_sharp_edges` | Yes | **NO** | **NO** | **NO** | **NO** | Yes | **NO** | Yes |
| `chamfer_error` | Yes | Yes | Yes | Yes | Yes | Yes | **NO** | Yes |

Pattern: Python bindings lag behind TypeScript. TS tests lag behind both. C++ tests are sometimes missing for geometry utilities.

---

## Checklist for Adding a New Feature

### C++ Core
- [ ] Create `include/trueform/<module>/<function>.hpp`
- [ ] Add to `include/trueform/<module>.hpp` with `// IWYU pragma: export`
- [ ] Add Catch2 test in `tests/<module>/`
- [ ] Add to `docs/content/cpp/2.modules/` docs

### Python Binding
- [ ] Create wrapper header: `python/include/trueform/python/<module>/<function>.hpp`
- [ ] Create binding C++ file per type combo: `python/src/<module>/<function>_<types>.cpp`
- [ ] Add `sources.cmake` entry for new `.cpp` files (and new headers to the module's `headers.cmake` / `HEADERS_<MOD>`)
- [ ] Register in `python/src/<module>.cpp`
- [ ] Create Python wrapper: `python/src/trueform/_<module>/<function>.py`
- [ ] Export from `python/src/trueform/__init__.py`
- [ ] Add pytest test: `python/tests/test_<function>.py` (parametrize dtypes)
- [ ] Add to `docs/content/py/2.modules/` docs

### TypeScript Binding
- [ ] Create binding C++ file: `typescript/cpp/src/<module>/<function>.cpp`
  - Implement `sync_<function>()` and `async_<function>()`
  - Register in `EMSCRIPTEN_BINDINGS` block
- [ ] Add sync wrapper: `typescript/src/<module>/sync.ts`
- [ ] Add async wrapper: `typescript/src/<module>/async.ts`
- [ ] Export from `typescript/src/manual.ts` (+ `src/async/index.ts` for the async twin)
- [ ] Add test in `typescript/tests/test_<module>.mjs` and register it in `tests/run.mjs`
- [ ] Add to `docs/content/ts/2.modules/` docs


---

## Adding a NEW Module (not just a feature)

The per-feature checklist assumes the module exists. A new module (e.g.
`csg/`) additionally needs:

**Python** — `python/src/<module>.cpp` with `register_<module>(m)`
creating the submodule; forward-decl header
`python/include/trueform/python/<module>.hpp`; `register_<module>` call in
`python/src/main.cpp`; `include(src/<module>/sources.cmake)` +
`${MODULE_<MOD>_SOURCES}` in `python/CMakeLists.txt`; the module's
`headers.cmake` included and `${HEADERS_<MOD>}` aggregated in
`python/include/headers.cmake`. Pure-python `_<module>/` packages are
copied automatically (no CMake edit).

**TypeScript** — the two `.cpp` files added to the
`add_executable(trueform_wasm ...)` list in `typescript/CMakeLists.txt`;
`src/<module>/{sync,async,index}.ts`; exports in `manual.ts` and
`async/index.ts`.

**Gotcha**: the Python build is pinned to one interpreter
(`CMakeCache.txt` → `Python_EXECUTABLE`). Running tests with a different
python gives a misleading "circular import: cannot import _trueform"
error — match the interpreter.

---

## Cross-Language Conventions (reference: the csg module)

- **Stateful "sealed engine" bindings**: expensive-build objects
  (e.g. `CsgGraph`) keep the native class fully opaque; the language
  class holds the user-facing state itself (the input mesh objects, the
  passed config) and exposes query methods. Lifetime rides the existing
  machinery (ndarray members / shared_ptr handles + FinalizationRegistry).
  Reference: `python/include/trueform/python/csg/csg_graph_impl.hpp`,
  `typescript/cpp/src/csg/csg_graph_impl.hpp`, and the facades in
  `python/src/trueform/_csg/` and `typescript/src/csg/`.
- **Expression trees cross as a flat postfix `int` program**: `id >= 0`
  pushes `op(id)`; `-1/-2/-3` pop two and push or/and/difference; `-4`
  pops one and pushes complement. Python builds it with operator dunders,
  TS with builder methods (`.or/.and/.sub/.not`); the C++ decoder is the
  single shared definition.
- **Flag naming mirrors the C++ tags**: `tf::return_source_ids` →
  `return_source_ids=True` / `{ returnSourceIds: true }`;
  `tf::return_index_map` likewise. Do not invent new names ("labels" is
  overloaded by region/domain labels).
- **No null placeholders in TS**: when an optional expression precedes an
  options object, accept the options in the expression slot and
  discriminate at runtime (`instanceof`/`typeof` — the `cleaned()` idiom;
  see `splitExprArgs` in `typescript/src/csg/sync.ts`).
- **Config enums cross as validated ints**: a C++ enum option (e.g.
  `tf::triangulation_type`) crosses every boundary as an `int` and is
  spelled per language at the facade — Python string kwarg with a
  `_MAP` + ValueError (`triangulation="refined_cdt"`), TS string-union
  option (`triangulation: "refinedCdt"`), `static_cast` at the single
  C++ conversion site. Options meaningless outside one surface stay in
  a surface-local options type, never in a shared one. Reference: the
  csg `triangulation` plumbing, mirrored by the arrangements.

---

## The Gate Rule

A change to a C++ surface that bindings call is NOT landed until the
binding builds and suites ran: the python extension
(`cmake --build build_python2 --target _trueform trueform_copy_python_files`
+ pytest) and the wasm module (`typescript: node build.mjs` + the test
harness). Implicit conversions can keep binding code COMPILING while
silently changing behavior — and signature changes can break extension
builds that nothing else exercises. Both happened on the same day.
