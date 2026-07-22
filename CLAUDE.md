# Trueform Claude Contract

This repository's architecture is straightforward; its execution style is not.
Conventional C++ and geometry advice is frequently slower or structurally wrong
for Trueform. Do not work from learned defaults when the repository provides a
specific pattern.

`AGENTS.md` is the authority. This file exists so Claude-based agents enter the
same instruction chain; it is deliberately not a second style guide or API
catalog.

## Mandatory read order

For any nontrivial C++ or performance-sensitive change, read:

1. `AGENTS.md`
2. `agents/working_method.md`
3. `agents/cpp_performance_philosophy.md`
4. `agents/cpp_execution_patterns.md`

Then read only the relevant reference:

- `agents/cpp_core_architecture.md` for core types and primitives;
- `agents/cpp_engineering_philosophy.md` for implementation conventions;
- `agents/cpp_modules.md` for module/API lookup;
- `agents/python_layer.md` for NumPy ownership and Python/TBB boundaries;
- `agents/typescript_layer.md` for WASM ownership and async dispatch;
- `agents/feature_lifecycle.md` for a public cross-language feature;
- `agents/documentation_architecture.md` for documentation work;
- `agents/usage_cpp.md`, `agents/usage_python.md`, or
  `agents/usage_typescript.md` for caller-facing composition and public usage.

Do not substitute this file for that reading. If a statement here conflicts
with `AGENTS.md`, `AGENTS.md` wins.

## How to investigate

Trace the actual producer-to-consumer pipeline before proposing a change:

```text
owning buffers
-> semantic ranges
-> cheap discovery
-> sort/count and compute offsets
-> independent blocks
-> expensive block-local work
-> ordered aggregation when carrier identity matters
-> dense remap/equivalence collapse
-> final ownership boundary
```

Read callees as well as named entry points. For performance work, identify:

- the carrier that owns the question;
- where parallel grain is manufactured;
- which offsets, positions, or dense IDs preserve identity;
- which facts have one authoritative producer;
- which barrier separates discovery from materialization;
- whether state belongs to a partition, block, or irregular tree traversal;
- which serial phase is intentionally cheaper than parallel coordination.

Prime exemplars include intersection graph construction, polygon intersections,
face regions, region triangulation, and the helpers they call. Copy their work
shape only when the new problem has the same carrier and ownership structure.

## Defaults that are wrong here

Do not reach first for:

- hash maps for bounded integer identities;
- thread-local containers for partitionable work;
- nested owning containers for jagged data;
- concurrent append when counts and offsets can assign disjoint output;
- locks or atomics where a phase boundary or structural separation works;
- coordinate equality for topological identity;
- recomputation of facts already produced by an earlier stage;
- parallel inner loops whose work is tiny, dependent, or stateful.

The normal alternatives are dense sentinel maps, flat buffers plus ranges,
offset-blocked carriers, sort/group, count-prefix-allocate, partition-carried
state, sequenced aggregation, and post-barrier equivalence collapse.

`local_value`, `local_buffer`, and `local_vector` are narrow escape hatches for
irregular callback-driven tree traversal where no sequential partition can
carry state. Existing use is not general precedent.

## Binding boundaries

- TypeScript TypedArray heap views borrow native WASM storage.
- TypeScript async workers capture owning native handles by value, never touch
  `emscripten::val`, and convert results during main-thread retrieval.
- Python wrappers retain NumPy owners and construct native ranges over their
  contiguous memory.
- Python TBB workers perform pure C++ work; NumPy/nanobind construction and
  cache commits happen on the GIL-owning calling thread after a barrier.
- Bindings preserve C++ labels, offsets, topology, and cache authority rather
  than deriving parallel implementations in the host language.

Read the complete binding contract before editing either surface.

## Working discipline

- Establish the correctness oracle before optimizing.
- Verify output identity before trusting timing.
- Use the narrowest relevant test/build gate first.
- Inspect current manifests and neighboring registrations instead of assuming a
  fixed binding file layout.
- Do not encode transient coverage reports, known-test counts, personal paths,
  stale build directories, or experiment results in instruction files.
- Do not generalize a rare mechanism without explaining the work shape and
  measuring it against the Trueform alternative.

Current build commands and validation expectations live in `AGENTS.md` and
`agents/feature_lifecycle.md`; do not duplicate them here.
