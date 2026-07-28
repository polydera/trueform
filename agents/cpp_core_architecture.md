# C++ Core Primitive Reference

The `core/` module supplies Trueform's type system, memory model, range
abstractions, policies, and parallel primitives. This is a factual lookup
reference, not the authority for algorithm design. Read `AGENTS.md` and follow
its **Read first** order exactly before using this reference.

---

## 1. Primitive Type Hierarchy

### 1.1 Base Layer: `pt_view`, `vec_view`, `pt`, `vec`

**Files**: `core/base/pt.hpp`, `core/base/vec.hpp`

The lowest level. Two families — points and vectors — each with view (non-owning) and value (owning) variants.

```
pt_view<T, Dims>         — mutable view (T* pointer)
pt_view<const T, Dims>   — read-only view (const T* pointer, assignment deleted)
pt<T, Dims>              — owning value (std::array<T, Dims>)

vec_view<T, Dims>        — mutable view
vec_view<const T, Dims>  — read-only view
vec<T, Dims>             — owning value
```

All expose:
```cpp
using element_type = T;          // or const T for const views
using value_type = T;
using coordinate_type = std::decay_t<T>;
using coordinate_dims = std::integral_constant<std::size_t, Dims>;
```

Key methods: `data()` returns pointer. Views store a raw `T*`, values store `std::array<T, Dims>`. Assignment between views copies element-by-element. Const views delete all assignment operators.

### 1.2 Policy Wrappers: `point_like`, `vector_like`

**Files**: `core/point_like.hpp`, `core/vector_like.hpp`

These are the public-facing types. They inherit from a Policy (one of the base types above) and add arithmetic, iteration, comparison, and type conversion.

```cpp
template <std::size_t Dims, typename Policy>
struct point_like : public Policy {
    using Policy::Policy;
    using Policy::operator=;
    // ...
};
```

**`point_like` provides**:
- `operator[]`, `begin()`, `end()`, `size()` (constexpr = Dims)
- `length2()`, `length()` (squared and euclidean)
- `as<U>()` — type conversion
- `as_vector()`, `as_vector_view()` — reinterpret as vector
- `point + vector → point`, `point - point → vector`, `point ± vector`
- Lexicographic comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`

**`vector_like` adds** scalar arithmetic: `vector * scalar`, `scalar * vector`, `vector / scalar`, unary `-`.

**Zero-cost**: `sizeof(point_like<3, pt<float, 3>>) == sizeof(std::array<float, 3>)`. No virtual dispatch, no indirection.

### 1.3 Convenience Aliases

```cpp
template <typename T, std::size_t Dims>
using point = point_like<Dims, core::pt<T, Dims>>;

// Usage:
tf::point<float, 3> p{1.0f, 2.0f, 3.0f};
```

View aliases follow the pattern `point_like<Dims, core::pt_view<T, Dims>>` — these are what iterators dereference to.

### 1.4 Segments and Polygons

**Files**: `core/segment.hpp`, `core/polygon.hpp`, `core/base/seg.hpp`, `core/base/poly.hpp`

Both are thin wrappers over ranges of points.

```cpp
template <std::size_t Dims, typename Policy>
struct segment : public Policy { /* forwards [], begin, end, size */ };
// static_size = 2

template <std::size_t Dims, typename Policy>
struct polygon : public Policy { /* forwards [], begin, end, size */ };
// static_size = V (compile-time) or dynamic_size
```

**Base storage types**:
- `core::seg<Policy>` inherits `assignable_range<2, Policy>` — always 2 points
- `core::poly<V, Policy>` inherits `assignable_range<V, Policy>` — V points (or dynamic)

**Factory functions**:
```cpp
tf::make_segment(range_of_2_points)
tf::make_segment(indices, points)            // indirect
tf::make_segment_between_points(pt0, pt1)

tf::make_polygon<V>(range_of_V_points)
tf::make_polygon(indices, points)            // indirect
```

### 1.5 Unwrap/Wrap Pattern

Every wrapper type supports:
```cpp
unwrap(const wrapper &w) -> const Policy &;            // strip wrapper
wrap_like(const wrapper &template, T&& policy) -> wrapper<...>;  // re-wrap
```

This enables generic algorithms that operate on the Policy level while preserving the outer wrapper type.

---

## 2. Type Traits

### 2.1 `coordinate_type<T>`

**File**: `core/coordinate_type.hpp`

Recursively extracts the scalar type:
1. If `T` is fundamental → `T`
2. If `T::coordinate_type` exists → use it
3. If `T::value_type` exists → recurse on `value_type`

Multi-type: `coordinate_type<T, Ts...>` = `std::common_type_t<...>`.

### 2.2 `coordinate_dims_v<T>`

**File**: `core/coordinate_dims.hpp`

Recursively extracts dimensionality:
1. Default → 1
2. If `T::coordinate_dims` exists → `T::coordinate_dims::value`
3. If `T::value_type` exists → recurse

### 2.3 `static_size_v<T>`

**File**: `core/static_size.hpp`

Returns compile-time element count, or `dynamic_size = size_t(-1)`:
- `std::array<T, N>` → N
- `segment<Dims, Policy>` → 2
- `polygon<Dims, Policy>` → depends on Policy
- `point_like<Dims, Policy>` → Dims
- `range<Iterator, N>` → N

---

## 3. Collection Types (Buffers and Views)

### 3.1 The Buffer/View Split

This is a foundational design principle. **Buffers own memory. Views reference it.**

| Owning (buffer) | Non-owning (view/range) | Element view |
|-----------------|------------------------|--------------|
| `points_buffer<T, Dims>` | `tf::points<Policy>` | `point_like<Dims, pt_view<T, Dims>>` |
| `polygons_buffer<Index, RealT, Dims, Ngon>` | `tf::polygons<Policy>` | `polygon<Dims, poly<Ngon, ...>>` |
| `segments_buffer<Index, RealT, Dims>` | `tf::segments<Policy>` | `segment<Dims, seg<...>>` |
| `vectors_buffer<T, Dims>` | `tf::vectors<Policy>` | `vector_like<Dims, vec_view<T, Dims>>` |
| `curves_buffer<Index, RealT, Dims>` | `tf::curves<Policy>` | polyline views |

The `.polygons()` / `.points()` / `.faces()` accessors on buffer types return the corresponding view.

### 3.2 Internal Storage

All buffers use **interleaved flat storage**:
- `points_buffer<float, 3>` stores `[x0,y0,z0, x1,y1,z1, ...]` in a single `tf::buffer<float>`
- `polygons_buffer<int, float, 3, 3>` stores faces in `blocked_buffer<int, 3>` (3 indices per triangle) and points in `points_buffer<float, 3>`
- Variable-size polygons use `offset_block_buffer<Index, Index>` for faces

### 3.3 The `form<Dims, Policy>` Base

**File**: `core/form.hpp`

```cpp
template <std::size_t Dims, typename Policy>
struct form : public Policy { /* unwrap/wrap support */ };
```

`tf::points`, `tf::polygons`, `tf::segments` all inherit from `form`. This enables policy composition — tagging a `tf::polygons` with a spatial tree produces a new `tf::polygons` with tree access, without copying data.

---

## 4. Memory Model

### 4.1 `tf::buffer<T>` — Primary Allocation Primitive

**File**: `core/buffer.hpp`

```cpp
template <typename T> class buffer {
    static_assert(std::is_trivially_destructible<T>::value, "Just use std::vector");
    std::unique_ptr<T[]> _data;
    T *_end, *_capacity;
};
```

**Key properties**:
- **Trivially destructible types only** — no constructor/destructor calls, just `memcpy` for moves
- **Uninitialized allocation** — `allocate(n)` doesn't zero memory (deliberate for performance)
- **Growth strategy**: `new_capacity = size() + max(size(), added)` (roughly doubling)
- **Raw pointer iterators**: `begin()` / `end()` return `T*`

**Methods**:
- `allocate(n)` — set size to n, grow capacity if needed (content undefined for new elements)
- `reallocate(n)` — grow to n preserving existing content
- `reserve(n)` — ensure capacity >= n without changing size
- `push_back(T)` — append one element, growing if needed
- `clear()` — set size to 0, keep capacity

### 4.2 Block Storage

- **`blocked_buffer<T, BlockSize>`** — flat `buffer<T>` accessed as fixed-size blocks. Iteration yields `range<T*, BlockSize>` views. Used for triangles (BlockSize=3), edges (BlockSize=2).
- **`offset_block_buffer<Index, T>`** — variable-length blocks via `buffer<Index>` offsets + `buffer<T>` data. Used for variable-size polygons, curve paths.

### 4.3 Arena-Local Storage — Traversal Escape Hatch

**Files**: `core/local_buffer.hpp`, `core/local_value.hpp`, `core/local_vector.hpp`, `core/cache_aligned_slot.hpp`

```cpp
template <typename T> struct alignas(128) cache_aligned_slot {
    T value;
    char padding[...];  // pad to 128-byte boundary
};
```

**`local_buffer<T>`**: One `buffer<T>` per TBB thread (indexed by `tbb::this_task_arena::current_thread_index()`). Methods: `push_back()`, `to_buffer()` (merge all threads), `clear()`, `reserve_all()`.

**`local_value<T>`**: One `T` per thread. Methods: `*local_val` (dereference to thread-local T), `aggregate(BinaryOp)` (combine all threads), `reset(val)`.

128-byte alignment prevents false sharing — each thread's slot is on a separate cache line.

These types are not the default parallel-state model. Use them only for
irregular parallel tree traversal whose callback/task API cannot provide a
stable block or local-state argument. Partitionable work carries state through
`parallel_for_each`, generation, or reduction primitives.

### 4.4 `small_vector<T, N>`

Typedef for LLVM's `SmallVector<T, N>`. Inline storage for N elements, heap overflow for more. Used for small, bounded collections (e.g., `small_vector<buffer<Index>, 10>` for per-mesh data when N meshes is small).

### 4.5 `none_t` — Zero-Cost Optional

**File**: `core/none.hpp`

```cpp
struct none_t {};
inline constexpr none_t none;
```

Used as default template parameter: `typename Index = tf::none_t`. Inside the function, `if constexpr (std::is_same_v<Index, tf::none_t>)` triggers type deduction or disables a feature. No runtime cost — the dead branch is eliminated at compile time.

### 4.6 Index Maps

**File**: `core/index_map.hpp`

```cpp
template <typename Range0, typename Range1> class index_map {
    Range0 _f;         // forward: old_id → new_id (or sentinel if removed)
    Range1 _inv_f;     // inverse: list of kept original IDs
};
```

**`index_map_buffer<Index>`**: Concrete version with `buffer<Index>` for both ranges.

**Factory**: `ids_to_index_map(ids, total_elements)` — builds forward + inverse maps from a list of selected IDs.
**Factory**: `mask_to_index_map(bool_mask)` — builds maps from a boolean mask.

---

## 5. Range Composition System

### 5.1 Philosophy

Data flows through lazy transformations. No intermediate buffers are allocated unless explicitly materialized. A `make_indirect_range(ids, faces)` doesn't copy — it creates a view that dereferences through the index array on each iteration.

### 5.2 The `range<Iterator, N>` Foundation

**File**: `core/range.hpp`

```cpp
// Fixed size: stores only begin, end = begin + N
template <typename Iterator, std::size_t N> class range;

// Dynamic size: stores begin and end
template <typename Iterator> class range<Iterator, dynamic_size>;
```

`make_range(begin, end)`, `make_range<N>(begin)`, `make_range(container)`.

### 5.3 Range Types

| Type | Factory | Semantics | Lazy? |
|------|---------|-----------|-------|
| `mapped_range` | `make_mapped_range(r, f)` | Transform: `f(*iter)` on each dereference | Yes |
| `indirect_range` | `make_indirect_range(ids, data)` | Gather: `data[*ids_iter]` | Yes |
| `block_indirect_range` | `make_block_indirect_range(blocks, data)` | Per-block gather: each block's indices resolved against data | Yes |
| `offset_block_range` | `make_offset_block_range(offsets, data)` | Grouped access: block i = `data[offsets[i]..offsets[i+1])` | Yes |
| `blocked_range` | `make_blocked_range<N>(r)` | Fixed-stride: group every N elements | Yes |
| `zip` | `tf::zip(r0, r1, ...)` | Parallel iteration: yields tuple of elements | Yes |
| `enumerate` | `tf::enumerate(r)` | `zip(sequence, r)` — index + value pairs | Yes |
| `sequence_range` | `make_sequence_range(n)` | Iota: `[0, 1, 2, ..., n-1]` | Yes |
| `slice` | `tf::slice(r, from, to)` | Subsequence `[from, to)` | Yes |
| `take` | `tf::take(r, n)` | First n elements | Yes |
| `drop` | `tf::drop(r, n)` | Skip first n elements | Yes |
| `constant_range` | `make_constant_range(val, n)` | Broadcasts single value n times | Yes |
| `slide_range` | `make_slide_range<W>(r)` | Sliding window of width W | Yes |
| `tagged_range` | `make_tagged_range(r)` | First element is tag, rest is range body | Yes |

All return `tf::range<SomeIterator, StaticSize>` — the static size propagates through composition via `tf::static_size_v`.

### 5.4 Composition Example

From `reindex/polygons.hpp` — remapping face vertex indices through an index map:

```cpp
tf::make_indirect_range(
    face_im.kept_ids(),                              // which faces to keep
    tf::make_block_indirect_range(polygons.faces(),   // face → [v0, v1, v2]
                                  point_im.f()))      // remap each vi → new_vi
```

This reads as: "For each kept face ID, look up the face's vertex indices, and remap each vertex through the point index map." Zero allocations — pure lazy composition.

### 5.5 Materialization

Lazy ranges become concrete via `tf::parallel_copy(range, buffer)`:

```cpp
tf::buffer<Index> output;
output.allocate(range.size());
tf::parallel_copy(range, output);
```

### 5.6 Iterator Architecture

All range iterators use a CRTP hierarchy:

```
forward_mapped_crtp<Derived, Iterator, DereferencePolicy>
  └── bidirectional_mapped_crtp<...>
        └── random_access_mapped_crtp<...>
```

Two dereference modes:
- **`mapped<Iter, Policy>`**: applies policy to `*iter` — used for element transforms
- **`iter_mapped<Iter, Policy>`**: applies policy to `iter` itself — used for stride/offset operations

Blocked/strided iterators use `stride_api<Derived, Handle>` which adjusts `++`/`--`/`+=` by a stride rather than 1.

---

## 6. Policy Composition System

### 6.1 The `tag()` / `operator|` Pattern

Policies are composed by wrapping. Each `tag_*` struct inherits from `Base` (the previous policy) and adds a new capability.

```cpp
// Tag a polygons view with a spatial tree
auto form = polygons | tf::tag(tree);
// form is: polygons<tag_tree<TreePolicy, original_policy>>

// Tag with tree AND face membership
auto form = polygons | tf::tag(tree) | tf::tag(membership);
// form is: polygons<tag_membership<..., tag_tree<..., original_policy>>>
```

### 6.2 Policy Structures

Each follows the same pattern:

```cpp
template <typename SpecificPolicy, typename Base>
struct tag_specific : Base {
    using Base::operator=;
    SpecificPolicy _specific;

    auto specific() const -> const SpecificPolicy& { return _specific; }
};
```

**Known tag policies** (from `core/policy/` and `spatial/policy/`):

| Policy | File | What it tags |
|--------|------|-------------|
| `tag_frame` | `core/policy/frame.hpp` | Coordinate transformation frame |
| `tag_tree` | `spatial/policy/tree.hpp` | AABB spatial tree |
| `tag_mod_tree` | `spatial/policy/tree.hpp` | Modifiable spatial tree |
| `tag_buffer` | `core/policy/buffer.hpp` | Attached buffer data |
| `tag_normals` | `core/policy/normals.hpp` | Per-face/vertex normals |
| `tag_ids` | `core/policy/ids.hpp` | Attached ID array |
| `tag_states` | `core/policy/states.hpp` | Per-element state data |

### 6.3 SFINAE Introspection

Each policy provides a detection trait:

```cpp
// In tag_tree.hpp:
auto has_tree(const tag_tree<...> *) -> std::true_type;
auto has_tree(const void *) -> std::false_type;

template <typename T>
inline constexpr bool has_tree_policy = decltype(
    policy::has_tree(static_cast<const std::decay_t<T>*>(nullptr)))::value;
```

Algorithms use `if constexpr (has_tree_policy<T>)` to conditionally access the tree — zero cost when absent.

### 6.4 `frame_of()` and `transformed()`

```cpp
template <typename T>
auto frame_of(const T& t) -> /* frame or identity */;

template <typename T, typename Frame>
auto transformed(const T& point, const Frame& frame) -> /* transformed point */;
```

If `T` has a frame policy, `frame_of` returns it. Otherwise returns identity. `transformed` applies the frame's transformation. Both are zero-cost for identity frames via `if constexpr`.

---

## 7. Parallel Algorithm Primitives

### 7.1 Philosophy

**Use the parallel vocabulary at the correct grain.** Unchecked primitives enter
their TBB implementation. Overloads taking `tf::checked` use the primitive's
small-workload serial fallback. The primitive owns that cutoff; call sites
still choose whether range length is a sound proxy for their actual kernel.

### 7.2 Algorithm Vocabulary

| Algorithm | File | TBB Primitive | Semantics |
|-----------|------|--------------|-----------|
| `parallel_for_each(r, f)` | `core/algorithm/parallel_for_each.hpp` | `tbb::parallel_for(blocked_range)` | Apply f to each element |
| `parallel_for(r, f)` | `core/algorithm/parallel_for.hpp` | `tbb::parallel_for(blocked_range)` | Apply f to subranges [begin, end) |
| `parallel_copy(in, out)` | `core/algorithm/parallel_copy.hpp` | `tbb::parallel_for(blocked_range<size_t>)` | Copy elements |
| `parallel_copy_blocked(in, out)` | `core/algorithm/parallel_copy_blocked.hpp` | via `parallel_for_each` | Copy per-block |
| `parallel_fill(r, val)` | `core/algorithm/parallel_fill.hpp` | via `parallel_for` | Fill with value |
| `parallel_iota(r, start)` | `core/algorithm/parallel_iota.hpp` | via `parallel_for` | Sequential values |
| `parallel_transform(in, out, f)` | `core/algorithm/parallel_transform.hpp` | `tbb::parallel_for(blocked_range<size_t>)` | Transform elements |

### 7.3 The `tf::checked` Tag

```cpp
struct checked_t {};
static constexpr checked_t checked;
```

Passed as extra argument to enable small-workload fallback:

```cpp
tf::parallel_for_each(range, func);              // enter TBB path
tf::parallel_for_each(range, func, tf::checked);  // use serial fallback when small
```

### 7.4 Local State in Parallel Work

**The rule: never construct scratch containers inside a per-element
lambda.** A `small_vector`/`buffer` built per element allocates and
destroys once per element; every parallel primitive has a form that
gives you per-chunk (per-TBB-task) local state constructed once and
reused across the chunk's elements. Pick by what the loop produces:

| Need | Primitive | Local state |
|--|--|--|
| Side effects only, scratch needed | `parallel_for_each(r, f, State{})` | `f(elem, state)`; `state` copied once per chunk — clear and reuse it per element |
| Variable-length output, order irrelevant | `generic_generate(r, out, f)` | `f(elem, buffer)`; per-chunk buffer, aggregate order unspecified |
| Variable-length output, input order required | `sequenced_generate(r, out, f)` | per-chunk buffer, appended in input-block order |
| Offsets + data blocks per element | `generate_offset_blocks(r, offsets, data, f)` | same shape, emits offset-block structure |
| Reduce to one result, order irrelevant | `blocked_reduce(r, init, local, task, agg)` | `local_t` per chunk, `agg` merges |
| Aggregation order creates structure | `blocked_reduce_sequenced_aggregate(r, init, local_t{}, task, agg)` | `task(chunk_range, local)` fills locals in parallel; `agg(local, ...)` runs in input-block order to preserve jagged alignment, construct offsets, or rebase correlated outputs |
| Irregular parallel tree callback with no state slot (`tf::search`, ray casts, dual-tree traversal) | `local_buffer<T>` / `local_vector<T>` / `local_value<T>` | one slot per TBB thread (cache-aligned), merge via `to_buffer()`. LAST RESORT: every partitionable algorithm above threads local state through |

```cpp
struct local_t { tf::small_vector<seg_t, 16> segs; };
tf::parallel_for_each(range, [&](auto &elem, local_t &local) {
    local.segs.clear();   // reuse capacity across the chunk
    ...
}, local_t{});
```

The pipeline builders (`build_loops`, `split_edges`,
`strip_base_loop_edges`) are the reference pattern: a `local_t` struct of
reusable buffers plus `blocked_reduce_sequenced_aggregate` when input position is
the identity of the output block. The primitive is not only a determinism gate;
ordered aggregation preserves implicit joins between jagged carriers and can
eliminate later sorting or lookup.

### 7.5 Grouping Utilities

**`compute_offsets(sorted_data, out_iter, start, compare)`** — scans sorted data, outputs group boundary offsets where `compare(current, next)` fails. Used in sort-then-group workflows.

**`ids_to_index_map(ids, total)`** — builds `index_map` (forward + inverse) from selected IDs.

**`mask_to_index_map(bool_mask)`** — builds `index_map` from boolean mask.

---

## 8. Key Invariants

1. **Views never outlive their data.** Buffer is the owner. Views (points, polygons, ranges) reference buffer memory. The caller is responsible for lifetime management.

2. **Static size propagates.** If you `make_blocked_range<3>(buffer)`, the resulting range has `static_size = 3`. If you `make_indirect_range(ids, data)`, it inherits the static size of `ids`. This enables compile-time optimizations throughout the chain.

3. **Policies compose, never copy data.** `polygons | tag(tree)` wraps the existing policy in a new layer. The points and faces stay where they are. The tree reference is added to the policy chain.

4. **Parallel algorithms default to TBB's automatic partitioner.** Two explicit
   tuning knobs exist: the primitive-owned serial fallback via `tf::checked`,
   and an explicit grain via `tf::grain(n)` (`core/grain.hpp`) accepted by
   `parallel_for_each` — use either only when the work shape justifies it.

5. **No virtual dispatch anywhere.** All polymorphism is at compile time via templates, policies, and `if constexpr`.

---

## 9. The Arrangement Pipeline (`cut/` + `csg/`)

Everything above is `core/`. The geometric pipeline sits on top of it,
and its center is **`tf::arrangement_graph`** (`cut/arrangement_graph.hpp`)
— the arrangement of a set of forms, everything below classification:
intersections, the intersection graph, the face-region structure, the
coplanar stacks (pairs + dead mask, detected once here), and the region
triangulation with its unified created-points table.

```
forms → intersections_between_polygons → intersection_graph
      → face_regions (region walks)     → coplanar stacks
      → region_triangulator (triangle-grain stream + created points)
      = arrangement_graph
      + classification (descriptor, inclusions, volumes, sheets)
      = csg_graph
```

### The pieces

- **`tf::arrangement_config`** (`cut/arrangement_config.hpp`) — every
  arrangement surface's parameter: `{intersect_config intersect,
  triangulation_type triangulation}`, implicitly constructible from
  either alone (and from `intersect_mode`). Default intersect =
  `primitives | resolve_crossing_contours`.
- **Storage policies** (`cut/dispatch/arrangement_range_policy.hpp`,
  `arrangement_pair_policy.hpp`) — the graph's ctor takes ONLY a
  policy. The range policy stores a homogeneous forms range + owned
  missing structures and rebuilds tagged views via `forms()`; the pair
  policy erases two DIFFERENT form types behind
  `apply_to_form(tag, f)`. The `make_arrangement_graph` factories own
  tag-completion: constexpr-branched build-only-what-is-missing
  (mirroring `dispatch::boolean`), so a fully tagged call site
  constructs with zero structure work.
- **`tf::face_regions`** (`cut/face_regions.hpp`) — the raw region
  structure: per-face region walks (boundary + holes) over intersection
  identities. Classification is REGION-grain.
- **`tf::cut::region_triangulator`** (`cut/impl/region_triangulator.hpp`) —
  owner of the exposed TRIANGLE-grain stream (`loops()` = one triangle
  each, contiguous per tag, promoted faces after each tag's structure
  loops) and of provenance the stream cannot carry: `merges()` (the
  weld table + `resolve()`), `original_edge_splits()`,
  `promoted_descriptors()`. Nothing downstream ever triangulates.
  `triangulation_type::refined_cdt` quality-refines through the same
  machinery (dyadic split parameters negotiated globally; boundaries
  watertight by construction).
- **`created_points()`** — the unified table on the exact integer
  lattice: intersection-graph points first, then everything the
  triangulation materialized. Created vertex ids index it directly;
  identity = `{tag, id}` pairs with created ids past the last tag.
- **Exact substrate** (`exact/`): `resolve_int_type` picks the lattice
  int from the input real; `vertex_converter` converts/deconverts;
  `make_kernel` builds the predicate kernel. Coordinates in the
  pipeline are lattice ints; float conversion happens at the edges.
- **`tf::csg_graph`** (`csg/csg_graph.hpp`) = an arrangement_graph plus
  the classification tier (arrangement descriptor, domain inclusions,
  volumes, sheet anchoring; machinery in `cut/arrangements/` and
  `cut/classification/`). Consumers: `make_csg_mesh` (boolean
  expressions), `make_csg_domains`, `make_outer_shell`,
  `make_intersection_curves(csg_graph)`.
