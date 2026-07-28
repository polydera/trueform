# Public Feature Lifecycle

Use this checklist only when adding or changing a public feature across C++,
Python, TypeScript/WASM, tests, and documentation. It describes obligations,
not a fixed file-per-instantiation layout. Binding layouts evolve; inspect the
nearest current feature and every relevant build manifest before editing.

Read `AGENTS.md` and follow its **Read first** order exactly. Then read
`documentation_architecture.md` before changing `docs/content/`, plus
`python_layer.md` and/or `typescript_layer.md` for the affected surfaces.

## 1. Define the semantic surface

Before creating files, write down:

- the authoritative C++ operation and carrier;
- template dimensions and coordinate/index types actually supported;
- whether output preserves static arity or is genuinely jagged;
- ownership and lifetime of returned buffers/views;
- optional policies, tags, configuration, and return-index-map variants;
- exact naming and defaults for C++, Python, and TypeScript;
- sync/async parity requirements;
- the correctness oracle shared across languages.

Do not extend a shared semantic merely to make one binding convenient. Surface
that design fork before implementation.

## 2. C++ core

- Add the implementation under `include/trueform/<module>/`.
- Add the public include to the module umbrella with
  `// IWYU pragma: export` when it belongs on the public surface.
- Follow the owning-buffer/non-owning-range split.
- Preserve static-size information and existing policies.
- Use the execution pattern appropriate to the work shape; do not copy the
  surface structure of a neighboring function while changing its carrier.
- Add Catch2 coverage under `tests/<module>/` using the existing local test
  organization and relevant type matrix.
- Validate the narrow C++ target before binding work.

## 3. Python surface

Inspect the nearest current binding with the same input and output carriers.
Python bindings use more than one instantiation layout; do not assume one `.cpp`
per type combination.

Required obligations:

- Add or update the nanobind wrapper under
  `python/include/trueform/python/<module>/` when a wrapper is needed.
- Add instantiations or registration sources using the module's current pattern.
- Update every source/header manifest that owns those files.
- Register the function or type in the correct native submodule.
- Add or update the Python facade and dtype/shape dispatch.
- Export it from the public Python surface.
- Preserve ndarray ownership and lifetime rules.
- Keep TBB worker phases free of Python/nanobind operations; construct NumPy
  results and commit Python-owned cache state on the GIL-owning thread.
- Add pytest coverage for supported dtype/dimension/arity combinations and
  failure behavior at the language boundary.
- Update Python documentation when the feature is public.

Naming normally drops a C++ `make_` prefix and remains `snake_case`, but verify
the neighboring public API before introducing a new transformation.

## 4. TypeScript/WASM surface

Inspect the nearest current binding with the same carrier and threading model.
Current bindings may use a shared `_impl.hpp` plus separate float32/float64
registration translation units. Do not assume one binding `.cpp`, float-only
support, or one universal registration shape.

Required obligations:

- Add or update embind implementation and dtype registrations using the current
  module pattern.
- Update `typescript/CMakeLists.txt` or the owning source manifest.
- Provide the synchronous wrapper when the operation is synchronous.
- Provide the async/dispatcher twin when the public module promises one.
- Preserve handle ownership, ndarray lifetime, and explicit `.delete()` behavior.
- Treat TypedArray heap views as borrowed, capture async native handles by value,
  and keep `emscripten::val` conversion on the main-thread retrieval phase.
- Export through the current manual and async surfaces.
- Add tests and register them in the current test runner.
- Update TypeScript documentation when the feature is public.

Naming normally drops C++ `make_` and converts to `camelCase`. Configuration
enums cross at one validated conversion site; do not duplicate interpretation
between sync and async facades.

## 5. Cross-language identity

Bindings should expose the same computation, not approximate rewrites of it.

- C++ remains the authority for geometry, topology, identity, and classification.
- Bindings convert storage and naming; they do not rederive algorithmic facts.
- Index maps, source IDs, tags, and optional result carriers preserve their C++
  meaning across languages.
- Stateful expensive-build objects remain sealed native engines. Language
  facades own user-facing state and lifetime handles.
- Optional flags mirror the C++ tags unless a language boundary requires a
  documented transformation.

## 6. Documentation

Follow `documentation_architecture.md` rather than assuming every feature needs
three parallel pages. Keep examples semantically aligned across languages while
using the idiomatic public surface of each one.

Document:

- accepted carriers and dtypes;
- sync/async availability;
- ownership or disposal requirements;
- return shapes and index-map semantics;
- exactness/tolerance behavior that users can observe;
- one minimal example whose assertions could fail under a broken binding.

## 7. Gates

`AGENTS.md` is the single command authority. Run its narrowest relevant gate
after each layer, then the complete affected stack. Use one fresh Python wheel
directory and the same `python -m pip` / `python -m pytest` interpreter. For
TypeScript/WASM, run the registered test harness as well as the build and type
check.

Do not hardcode personal build directories in documentation. Verify the binary
was rebuilt and count build errors and warnings before trusting a filtered log.

## 8. Landing checklist

- [ ] C++ public include and implementation
- [ ] C++ correctness tests
- [ ] Python native registration, facade, exports, and tests if exposed
- [ ] TypeScript dtype registrations, sync/async facades, exports, and tests if exposed
- [ ] Build manifests updated for every new source/header
- [ ] Documentation updated according to the site architecture
- [ ] Cross-language defaults and return semantics match
- [ ] Output identity proven before performance comparison
- [ ] Relevant C++, Python, and WASM builds pass from fresh artifacts
- [ ] No transient coverage status, local path, or experiment result added to this file

The checklist is intentionally structural. Current code and build manifests
determine registration and layout facts. Documented ownership, namespace,
execution, and style laws still govern; surface a conflict rather than treating
an isolated neighboring exception as precedent.
