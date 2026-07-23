# trueform agent contract

Trueform is a header-only C++17 geometry library built for exact, real-time
processing at million-polygon scale, including interactive TypeScript/WASM
booleans. Its architecture is not unusually complicated. Its execution style
is unusual, and conventional C++ advice is often slower here.

This file is not a repository tour. It tells an agent how to reason before
changing Trueform.

## Read first

For any nontrivial core change, read in this order:

1. `agents/working_method.md` — investigation, debugging, proof, and landing.
2. `agents/cpp_performance_philosophy.md` — the governing design laws.
3. `agents/cpp_execution_patterns.md` — the concrete phase shapes and the
   primitives that implement them.

Then read only the task-specific reference:

- `agents/cpp_core_architecture.md` for types, ranges, policies, and primitives.
- `agents/cpp_engineering_philosophy.md` for implementation and portability.
- `agents/python_layer.md` or `agents/typescript_layer.md` for bindings.
- `agents/csg_pipeline_debugging.md` for CSG correctness investigations.
- `agents/usage_cpp.md` for caller-facing C++ composition, ownership, tagging,
  invalidation, and choosing the right materialization level.
- `agents/feature_lifecycle.md` for a cross-language public feature.
- `agents/documentation_architecture.md` for documentation work.

The `usage_*` files describe how callers compose the public API;
`cpp_modules.md` is API lookup. They are not implementation strategy. For core
changes, inspect the implementation and its callers after reading the execution
documents.

## The execution model

The recurring Trueform pipeline is:

```text
flat owning buffers
-> semantic non-owning ranges
-> cheap parallel discovery
-> parallel sort or parallel counts
-> compute offsets / prefix once
-> independent contiguous blocks
-> expensive block-local work in parallel
-> ordered jagged transformation when positional identity matters
-> dense remap or equivalence collapse
-> materialize once at the ownership boundary
```

The expensive parallel grain often does not exist initially. Manufacture it:

```text
generate records without ordering constraints
-> parallel-sort by the key that defines shared work
-> compute offsets at key boundaries
-> process the resulting groups independently
```

Offsets are not merely storage metadata. An offset-block index is simultaneously
a random-access group ID, a parallel work unit, and often the implicit join key
between pipeline stages.

When jagged carrier `A[i]` produces jagged carrier `B[i]`, preserve `i` with
`tf::generate_offset_blocks` or a
`tf::blocked_reduce_sequenced_aggregate` that constructs aligned offsets. The
contents may grow, split, merge, or disappear; the block position continues to
identify the face, loop, region, or other carrier without another map or join.
Use `tf::sequenced_generate` when flat output order matters but per-input block
identity does not.

## Required reasoning before hot-path changes

Answer these questions before proposing machinery:

1. What carrier owns the question: form, tree leaf, face, loop, region, edge,
   component, triangle, or block?
2. Can cheap unordered records be sorted into the expensive independent grain?
3. Can counts plus one prefix produce exact disjoint output ranges?
4. Can block position preserve identity through the next jagged stage?
5. Is the identity space bounded and therefore directly addressable?
6. Can a dense sentinel map, generation/watermark, CSR, or sorted sparse table
   replace associative lookup?
7. Can workers propose facts and assign global IDs only after a barrier?
8. Which producer is authoritative for each fact?
9. Can existing output be remapped, compacted, or substituted instead of
   recomputed?
10. What is the measured threshold below which the serial kernel is faster?

## Parallel-state rule

Partitionable work carries state through the partitioning primitive:

- `tf::parallel_for_each(..., State{})` for side effects with reusable scratch.
- `tf::generic_generate` for unordered variable output.
- `tf::sequenced_generate` for ordered variable output.
- `tf::generate_offset_blocks` for index-preserving jagged output.
- `tf::blocked_reduce` for partitioned work plus serial aggregation.
- `tf::blocked_reduce_sequenced_aggregate` when aggregation order is structural.

Within each block, use tight serial loops and reuse local capacity.

`tf::local_value`, `tf::local_buffer`, and `tf::local_vector` are escape hatches
for irregular callback/task traversal, especially parallel tree search, where
the API cannot expose a stable sequential partition or carry block state. They
are not the normal variable-output mechanism.

## Representation rules

- Represent geometry with `point`, `vector`, `unit_vector`, segments, polygons,
  and their view forms. Raw arrays/pointers are storage and interop boundaries,
  not semantic geometry carried through an algorithm.
- Own data in flat buffers; express grouping and transformation as ranges.
- Preserve fixed arity with `blocked_buffer`/`blocked_range`.
- Represent genuinely jagged data with flat data plus offsets.
- Sort IDs or compact records, not heavyweight geometry.
- Flatten tagged/local identities into one bounded integer space when possible.
- Dense integer identity uses buffers and sentinels.
- Repeated local use of a dense global keyspace uses generations or watermarks,
  not an O(global-size) clear per group.
- Rare global facts use sorted sparse tables with binary search.
- Hash maps are almost never the answer. A sparse/dynamic per-query traversal
  set or arbitrary/composite key can justify one when direct addressing would
  allocate or clear a disproportionate global domain. Require a matching
  Trueform work shape and measurement; bounded reusable identity remains dense.
- Topological identity and coordinate equality are different facts.

## Phase rules

- Discover in parallel; materialize shared identity once; consume afterward.
- Let parallel traversal over-segment when cheap; emit equivalence pairs and
  collapse them after the barrier with a dense/sparse equivalence-class map.
- Parallel topology mutation requires structural separation, not locks: freeze
  separators, mutate independent interiors, then process the frontier.
- Recovery is a wave: discover -> materialize -> mark dirty -> retry.
- Stable aggregation is purchased when it preserves carrier alignment, creates
  query-ready offsets, or removes a later sort. It is not required everywhere.
- Prefix scans, adjacent sweeps, union-find, tiny sorts, leaf kernels, graph
  walks, and stateful triangulation may deliberately be serial.

## Rare mechanisms require proof

Do not introduce any of the following merely because they are conventional:

- a hash map for bounded integer identities;
- thread-local state for partitionable input;
- a mutex or atomic instead of a phase boundary or structural separation;
- nested owning containers instead of flat data plus ranges;
- concurrent append when counts/offsets can assign disjoint writes;
- coordinate comparison for topology;
- a second derivation of a producer-owned fact;
- rerunning an expensive producer after a remap or merge;
- parallelizing a small dependent inner kernel.

An existing narrow exception is not a general precedent. Explain the work shape
that makes the exception necessary and measure it.

## Binding boundaries

Bindings preserve the C++ computation and make ownership explicit. They do not
rederive geometry, topology, labels, offsets, or cache state.

- TypeScript TypedArray views into WASM are borrowed. Keep a native owner alive,
  reacquire views after possible memory growth, and explicitly dispose large
  native objects.
- TypeScript async workers capture native handles by value, never touch
  `emscripten::val`, and convert results only during main-thread `retrieve`.
  Shared lifetime is not permission for concurrent mutation.
- Python wrappers retain input ndarrays and build ranges over their memory.
  Normalize contiguous storage at the boundary when native code walks linearly.
- Python calls are synchronous but may use oneTBB internally. Worker phases are
  pure C++; NumPy/nanobind construction and cache commits happen on the
  GIL-owning calling thread after a barrier.
- Native result buffers move across language boundaries with matching ownership
  and allocators. Never expose a borrowed view without retaining its owner.

Read `agents/typescript_layer.md` or `agents/python_layer.md` before changing a
binding. Their ownership and concurrency contracts outrank neighboring syntax.

## Build and validation

### C++ tests

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

### TypeScript/WASM

```bash
cd typescript
npm run build
npm run typecheck
```

`typescript/build.mjs` configures the repository root with
`TF_BUILD_TYPESCRIPT=ON` and builds `trueform_wasm`.

### Benchmarks

```bash
cmake -B build -DTF_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target benchmarks
```

Use the narrowest relevant gate first, then broaden. Do not trust timing until
outputs are proven identical and the binary is known to be fresh.
