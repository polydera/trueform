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
| 15. TS `index.ts` | `typescript/src/index.ts` | Public export | `export { func } from './<module>'` |
| 16. TS Test | `typescript/tests/test_<module>.mjs` | assert-based | Manual lifecycle (`.delete()`) |
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
- [ ] Add `sources.cmake` entry for new `.cpp` files
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
- [ ] Export from `typescript/src/index.ts`
- [ ] Add test in `typescript/tests/test_<module>.mjs`
- [ ] Add to `docs/content/ts/2.modules/` docs
