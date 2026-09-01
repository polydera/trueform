# trueform agent contract

Trueform is a header-only C++17 geometry library built for exact, real-time
processing at million-polygon scale, including interactive TypeScript/WASM
booleans. Its architecture is not unusually complicated. Its execution style
is unusual, and conventional C++ advice is often slower here.

This file is not a repository tour. It tells an agent how to reason before
changing Trueform.

## How to think about this codebase

Trueform is arrays being shuffled — sorted, partitioned, copied — and nothing
else. A mesh does not exist. What exists is flat buffers of numbers wrapped in
semantic views (blocked, offset-block, indirect, mapped ranges; policy-tagged
forms) so a caller iterates polygons and topology while the implementation
stays literal: numbers moving between arrays. Hold both sides of that duality
deliberately — the semantic view for the user, the flat pass for the machine —
and never let an abstract noun (a graph, a mesh, a database) become a real
object in the design.

Each component is exactly one thing and nothing more; specialization is where
the speed comes from. The recurring move is: precompute in parallel, compute
offsets, process in parallel. Random access is built from tickets — an index
into another array, or a -1 sentinel for "nothing there". When a problem
splits into states (cut/uncut, dirty/clean, failed/passed), solve each state
on its own structure and merge the solutions, never the structures. The
average path pays nothing for machinery only the exceptional path uses.

When in doubt, ask: how would this be done by hand — take this, move it over
there? Then find the sort, the offsets, and the tickets that make that literal
motion fast. Do not ask how the world does it. Trueform is by now large enough
that a similar problem has almost certainly been solved in it already; find
that implementation and read it before designing anything. It will show how
this class of problem reduces to a reshuffling under this philosophy: what
gets precomputed, where the offsets fall, which tickets make the answers
random-accessible, how the problem was divided into small specific problems
each worth optimizing on its own, and how their answers were merged.

## Acceptance standard

Review every C++ change on three coequal axes: correctness, performance, and
unmistakable Trueform style. A change is unfinished if it fails any one.

Style here is a hard requirement, not a preference. The execution shapes in
this contract are what produce the performance and the robustness, so code that
passes its tests but does not look like the module it joins is not finished —
it has to be indistinguishable from the code around it. Working code in a
foreign shape is a defect report, not a contribution.

Check your own C++ against this list before presenting it:

- every new header opens with the copyright/license block copied verbatim from
  a neighboring header, and directly includes every symbol it uses;
- every parallel loop runs at the largest independent carrier through a house
  primitive — the parallel width is the carrier count, never the operand, tag,
  or form count;
- sorts move compact records by value; they do not sort indices compared
  through an indirect lookup into another array;
- no fact has two producers. When a refactor turns a member into a parameter,
  the member stays the authority and every call site passes it;
- local state structs carried by a parallel primitive are value-initialized;
- new work added to a hot path has a consumer and a measurement, or it does not
  land;
- a first-level module directory contains only the user-facing `tf::` surface —
  anything in a nested namespace lives in a nested directory;
- classes stay thin: each self-contained step is a free function in the
  module's nested namespace, one operation per header.

## Read first

For any nontrivial core change, read in this order:

1. `agents/working_method.md` — investigation, debugging, proof, and landing.
2. `agents/cpp_performance_philosophy.md` — the governing design laws.
3. `agents/cpp_execution_patterns.md` — the concrete phase shapes and the
   primitives that implement them.
4. `agents/cpp_engineering_philosophy.md` — implementation style, memory
   discipline, and portability.

Then read only the task-specific reference:

- `agents/cpp_core_architecture.md` for types, ranges, policies, and primitives.
- `agents/cpp_modules.md` for current module and API lookup.
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

Pass `tf::checked` when the iterated range length is a sound proxy for total
work and the small-range case should stay serial. It preserves one bulk
algorithm while selecting the primitive's serial fallback; do not replace it
with a second hand-written size branch. Do not use it mechanically: a short
range should remain parallel when each element costs more than the threading
overhead, as with large polygons, deep blocks, or other expensive tasks.

`tf::local_value`, `tf::local_buffer`, and `tf::local_vector` are reserved for
irregular parallel tree traversal whose callback/task API cannot expose a
stable sequential partition or carry block state. All partitionable work
carries state through its parallel primitive.

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

## Timeless code and comments

- Prefer self-documenting names and structure. Do not comment code that already
  states its purpose clearly.
- Comments explain only a non-obvious reason, invariant, ownership rule, or
  contract. They do not narrate the implementation.
- Names, branches, structure, and comments describe only the present mechanism
  and contract. Git history and task records own how code was developed,
  replaced, benchmarked, or debugged.

## Implementation by subtraction

A change is not finished when the new mechanism works; it is finished
when everything the mechanism replaced is gone. The default failure
mode of a model-assisted change is additive: new code layered beside
old code, adapters between them, special cases guarding the seam.
Trueform features land by removal — when new structure subsumes old
machinery, the old machinery is deleted in the same change, and
consumers are simplified to the stronger invariant instead of being
taught to tolerate both.

- Before adding a mechanism, name the existing mechanism it makes
  unnecessary. If the answer is "nothing", question the design.
- After landing one, sweep for what it obsoleted: derivations that
  became identities, resolutions that became no-ops, special cases the
  new invariant makes impossible. Delete them. A "harmless"
  belt-and-braces leftover is a second producer of a fact, and that is
  a defect, not caution.
- This does not conflict with porting fidelity. A port reproduces a
  reference mechanism whole because the reference is the authority on
  its own behavior. Subtraction acts at the architecture level: when a
  new invariant erases the REASON a ported mechanism existed, the
  mechanism goes with it — in the same change that erased the reason.
- Removal is not feature loss. The features stay; the structure
  carrying them shrinks. Remove until there is nothing left to remove.

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
- Every dispatched TypeScript context reaches retrieval and erasure on every
  terminal path, including when result conversion throws. Failure erases the
  context without converting an absent result.
- Python wrappers retain input ndarrays and build ranges over their memory.
  Normalize contiguous storage at the boundary when native code walks linearly.
- Python calls enter through nanobind and keep the GIL on the bound calling
  thread while oneTBB may run pure C++ workers internally. NumPy/nanobind
  construction and mutation of Python-owned wrapper or cache state happen on
  that calling thread after a barrier. Pure native caches follow their own
  ownership and synchronization contract.
- Native result buffers move across language boundaries with matching ownership
  and allocators. Never expose a borrowed view without retaining its owner.

Read `agents/typescript_layer.md` or `agents/python_layer.md` before changing a
binding. Their ownership and concurrency contracts outrank neighboring syntax.

## Build and validation

### C++ tests

```bash
cmake -B build -DTF_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target trueform_tests
ctest --test-dir build --output-on-failure
```

### Writing tests

A test never instantiates the arrangement or csg pipeline on a type of its
own. It builds through the compiled builder tier in `tests/common/` —
`tf::test::build_self_csg_graph`, `build_pair_csg_graph`,
`build_range_csg_graph`, `build_self_arrangement`, `build_pair_arrangement`,
`build_range_arrangement`, `mesh_arrangements_of`, `polygon_arrangements_of`,
`boolean_of` — and reads through the compiled readers — `csg_mesh_of`,
`csg_mesh_with_source_ids_of`, `csg_mesh_with_index_map_of`,
`csg_domains_of`, `outer_shell_of`, `arrangement_mesh_of`,
`arrangement_curves_of`. Each is compiled once per (index, real, arity)
combination in `tests/builders/`, so a test TU carries no build kernel and
no reader. Operands are assembled one way (`tests/common/tagged_operand.hpp`:
the canonical tag order, the frame always present), because every
container, tag order or arity a test invents is a new specialization of the
whole pipeline. A public entry shape — a `std::vector` of forms, a C array,
a pre-tagged operand, the single-form overload, a heterogeneous pair — is
exercised once, in its suite's entry-shape test, not in every case. A new
type combination is a new TU in `tests/builders/`, never an instantiation
inside a test.

Test targets compile at `-O1` in unity batches of four, so a helper in an
anonymous namespace needs a name that is unique within its suite, and a
test never depends on its own optimisation level.

### Python

```bash
rm -rf build/python-wheel
CMAKE_BUILD_PARALLEL_LEVEL=8 python -m pip wheel . --no-deps -w build/python-wheel
python -m pip install --force-reinstall --no-deps build/python-wheel/trueform-*.whl
python -m pytest python/tests
```

### TypeScript/WASM

```bash
cd typescript
npm run build
npm run typecheck
node tests/run.mjs
```

`typescript/build.mjs` configures the repository root with
`TF_BUILD_TYPESCRIPT=ON` and builds `trueform_wasm`.

### Benchmarks

```bash
cmake -B build -DTF_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target benchmarks
```

Use the narrowest relevant gate first, then broaden. Do not trust timing until
outputs are proven identical and the binary is known to be fresh. Run one timed
benchmark process at a time; fully parallel benchmarks occupy the machine, so
overlapping processes invalidate timing.
