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
struct checked_t {
  unsigned long serial_below = 1000;
  constexpr auto operator()(unsigned long n) const -> checked_t;
};
static constexpr checked_t checked;
```

Passed as extra argument to enable small-workload fallback. The tag carries its
own cutoff, so `tf::checked(n)` states it for a call site whose per-element work
sits far from the convention's assumption:

```cpp
tf::parallel_for_each(range, func);                    // enter TBB path
tf::parallel_for_each(range, func, tf::checked);       // serial below 1000
tf::parallel_for_each(range, func, tf::checked(64));   // serial below 64
```

The tag is uniform across the vocabulary: `parallel_for`, `parallel_for_each`,
`parallel_transform`, `parallel_contains`, `reduce`, `blocked_reduce`,
`blocked_reduce_sequenced_aggregate`, `generic_generate`, `sequenced_generate`
and `generate_offset_blocks` all take it, each selecting its own serial kernel.

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
   tuning knobs exist: the serial fallback via `tf::checked` / `tf::checked(n)`,
   taken by the whole for/transform/reduce/generate vocabulary, and an explicit
   minimum chunk via `tf::grain(n)` (`core/grain.hpp`), taken by
   `parallel_for_each` alone — use either only when the work shape justifies it.

5. **No virtual dispatch anywhere.** All polymorphism is at compile time via templates, policies, and `if constexpr`.

---

## 9. The Arrangement Pipeline (`intersect/` + `arrangement/` + `csg/`)

Everything above is `core/`. The geometric pipeline sits on top of it,
and its center is **`tf::arrangement_graph`**
(`arrangement/arrangement_graph.hpp`) — the arrangement of a set of
forms, everything below classification: the intersection identity, the
per-plane arrangement, the coplanar stacks, the cells and their fences,
and the unified created-points table.

```
forms
  → tf::polygon_intersections                exact intersection identity
  → tf::intersect::graph::local_arrangement  plane graph + split tables
  → tf::arrangement::plane_arrangement       one triangulation per plane carrier,
      over tf::arrangement::plane_world<P>   built on the prepared world
  → cells → piece incidence → piece fences   the arrangement's own structure
  = tf::arrangement_graph                    owns the identity substrate both
                                             tiers state into
  + classification (components, radial fans, descriptor, domains)
  = tf::csg_graph
```

The graph's exposed currency is one slot per face of every form,
contiguous per tag, an uncut face keeping its descriptor row with an
empty triangle span; the exposed triangles follow in tag-major face
order with their corners in the stream's vertex language.

### The pieces

- **`tf::arrangement_config`** (`arrangement/arrangement_config.hpp`) —
  every arrangement surface's parameter: `{intersect_config intersect,
  triangulation_type triangulation}`, implicitly constructible from
  either alone (and from `intersect_mode`). Default intersect =
  `intersect_config{}` (`{mode, tolerance}`; `primitives | within` is the
  canonical mode spelling): contour crossings resolve unconditionally —
  between classes wherever a third tag can make such a pair, within one class wherever the run is `within` — and under `sos` no
  contact is ever coplanar, so coplanar walls do not pool and the domains
  they would have separated stay joined.
- **Storage policies** (`arrangement/policy/arrangement_range_policy.hpp`,
  `arrangement_pair_policy.hpp`) — the graph's ctor takes ONLY a
  policy. `tf::arrangement::arrangement_range_policy` stores a homogeneous
  forms range + owned missing structures and rebuilds tagged views via
  `forms()`; `tf::arrangement::arrangement_pair_policy` erases two DIFFERENT
  form types behind `apply_to_form(tag, f)`. The
  `tf::make_arrangement_graph` factories own tag-completion:
  constexpr-branched build-only-what-is-missing (mirroring
  `dispatch::boolean`), so a fully tagged call site constructs with
  zero structure work.
- **`tf::polygon_intersections`**
  (`intersect/polygon_intersections.hpp`) — the
  intersections-between-polygons of the library and the sole producer
  of point identity. `build` takes one form (self records implied), two
  forms, or a range of forms; arity plus `tf::intersect_config` decide
  between/within, and there are no alias types. A record's `id` is a
  canonical point NAME, not a slot in a coordinate table: below
  `n_vertex_points()` it is an original vertex (`vertex_anchor`), above
  it an exact parameter class on an original edge (`home_edge` +
  `exact_parameter`). No coordinate is computed or kept here.
- **`tf::intersect::graph::local_arrangement`**
  (`intersect/graph/local_arrangement.hpp`) — owns the intersections,
  the `plane_graph` (planes, frames, members, face descriptors,
  windings; THE PLANE IDENTITY IS THE NAME — `name_plane_carriers`
  gives every cut face the canonical name of the exact plane it stands
  on, read through the same positions the pipeline reads, and equal
  names are one carrier whatever the tags, with a collinear face's void
  name equal to nothing; the names live in the identity substrate,
  `identity_substrate::plane_name` answering every plane either tier
  holds) and the definition tables the splits are applied to. THE LINE
  LAW (`make_plane_face_frame`): a face's carrier is the support of its
  corners (`plane_face_support`) and nothing else, for every tier that
  founds a plane on a face; its name and its frame are two readings of
  that one support, so a void name and a line frame are one verdict, and
  a face whose corners are collinear is a line carrier whatever its
  definitions hold.
  Its `build` orchestrates the module's free functions:
  `detect_plane_crossings`, `close_plane_classes`,
  `collapse_plane_identities`, `order_plane_splits`,
  `discover_uncut_entrants`, `state_uncut_entrants`,
  `seat_entrant_planes`, `respan_plane_defs`, `seat_entrant_plane_edges`,
  `fuse_plane_defs`. An entrant is named by the same law, and THE
  SEATING LAW (`seat_named_planes`) is the one seater of every tier that
  admits faces — the local arrangement's entrants
  (`seat_entrant_planes`), the plane arrangement's promotion
  (`seat_plane_promoted_faces`) and the carve: a name a standing plane
  holds seats the face on that plane as one more member of the stack,
  equal names among the rest found one plane numbered past the standing
  ones in the order of their smallest face, a void name is a plane of its
  own, and the entrant's definitions move onto its plane's block.
  The product is the post-split
  definition table, canon-major with a rebuilt plane CSR — no splits
  side table — plus the entrance tables (the source faces the cut world
  never named: `entrants`, `entrant_descriptors`, `entrant_planes` — the
  frames of the planes entrants founded — and `entrant_orientations`,
  each in its seated plane's frame); the rewrite rows the identity gate
  absorbed are written into the substrate's local tables. THE GATE
  (`collapse_plane_identities`) compares the
  lattice position of every identity the cut world names — every
  definition endpoint, every class, every landing (a created point
  rounded onto its own carrier's end) and, when a band placed the
  inputs, every original vertex — and whatever occupies one position is
  one coincidence of THE ELECTION
  (`intersect/graph/elect_identity_components.hpp` —
  `elect_identity_component` for one coincidence,
  `elect_identity_components` over sparse relations — the one law every
  tier's identity statement is made by: the gate's, the plane waves'
  closure, the Steiner election, the carve's and the census's). The rank
  leads the key — `0` a standing created identity, `1` a fresh statement,
  `2` an original — so the smallest key speaks the coincidence: a standing
  created identity survives, otherwise the coincidence mints, and
  originals never survive. Which candidates one coincidence joins is the
  caller's statement. The gate is the build's writer of the local layer;
  the census and the carve are its others. THE
  ENTRANCE LAW, one mechanism for both tiers
  (`discover_uncut_entrants` = `discover_weld_entrants` +
  `discover_split_entrants`, filtered only by the asking tier's own
  "the world names this face" mask): an uncut face is emitted from the
  source mesh verbatim and names only its own vertices and whole edges,
  so the moment a tier retires an original vertex or splits an original
  edge, every source face holding that feature enters the cut world,
  found through the source mesh's vertex membership and edge link. A
  collapsed face enters as a line carrier and emits nothing; its
  neighbours are stitched on the survivor. `assert_promotion_is_complete`
  states the invariant under `tf::core::verification`.
- **The carve** (`arrangement/reseat/`, `tf::arrangement::reseat`) —
  when the gate fuses identities onto a position some carrier of the
  fused identity does not hold, that carrier is cut at the position
  inside the one build (`carve_world`). The carve's pieces, the faces
  it reaches and their neighbours form a delta world. The carve states
  only what it creates: a corner the delta holds speaks the world's own
  name for it — its merged name, or its identified class's anchor — and
  is never elected again, so a coincidence the world holds stays the
  gate's; a seat or a class the delta created speaks the created identity
  standing at its lattice position, and only where none stands does it
  mint, the originals there retiring into the mint (the gate's survivor
  rule). Every face holding a retired original enters before the carve
  is stated. The statement lands in the local arrangement before any
  plane tier reads it
  (`restate_carve`): mints as direct created identities, welds as merge
  rows (an identity only the replaced rows named is dead by definition,
  stated nowhere), pieces in the entrance tables with
  their source sides (`source_side`), retired faces out of their planes' live members, and
  the rows merged in key order in the entrance's own shape
  (`restate_plane_defs`). A carve that cannot be stated is refused in
  the census (`reseat_refusal`) and the world stands as built. A plane
  diff over a carve carries a block only as far as the carve left it: a
  local row of a face the carve replaced drops, the rows the carve
  founded on that carrier join the block by reference, and a plane-tier
  identity only the dropped rows named is left unreferenced — dead by
  definition, stated nowhere and derived only by the exposure accounting
  (`account_created_identities`).
- **THE TWO POLES AND THE WAVE'S GRAIN LAWS.** The engine serves two polar
  policies — the LA-backed boolean world (few dense carriers, real tables)
  and the virtual mesh world (`arrangement/mesh/`, not exported: one face is
  one carrier, the world answers arithmetically from the polygon and
  materializes its tables ONCE, at the barrier the first refusal or weld
  reaches — `close_plane_lazy_round`, where the canonical extent is frozen
  as its FIRST statement). The world's contract is stated once, by grain,
  in `plane_world.hpp`; the wave's laws are stated once in
  `plane_arrangement.hpp`'s class doc: a recovery round's cost is
  proportional to the dirty set; porting routes changed groups only; AN
  UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR BOTH CARRIERS; carrier
  lookup is the identity's own membership, built on demand
  (`carriers_of_flat`, compile-time optional). Per-triangle classification
  sidecars (`slot_parents`, `coplanar_of`, `stacked`) exist only under
  `record_triangle_arrangement()`, the request the classification consumer
  asks; a carrier whose prepared constraint set is one simple closed ring
  of 3-4 boundary sides fans without a CDT (`find_plane_carrier_fan` — the
  guard is the exact turn), on the boolean path too, soundly: a later
  statement re-enters it through the wave.
- **`tf::arrangement::identity_substrate`**
  (`arrangement/identity/identity_substrate.hpp`) — the one authority
  over an arrangement's identities, of its points AND of its planes. AN
  IDENTITY IS MINTED, WELDED OR NAMED ONLY THROUGH IT: its two
  layers are private — `L` (`local_identity_tables`: the name statements,
  classes, created classes, direct points, the closed rewrite rows, and
  the name of every plane the local tier holds) and `P`
  (`plane_identity_tables`: the waves' created classes, names and
  parameters, direct points, closed rewrite rows, and the names of the
  planes the promotion founded; neither layer states a death) — every table reads through
  it as it stands (`local_merges()`, `plane_merges()`, …), and every
  statement enters through a verb that states its law at the call: the
  local tier's `clear_local`, `name_local_planes`, `seat_local_planes`,
  `seat_local_entrant_planes`, `state_local_names`, `close_local_classes`,
  `collapse_local_identities` (the gate), `mint_local_points`,
  `weld_local`; the plane tier's `clear_plane`,
  `mint_plane_classes`, `mint_plane_points` with `state_plane_point`,
  `weld_plane`, `seat_promoted_planes`, `seat_founded_planes`,
  `carry_plane`. Each verb writes its table where
  the pre-authority site wrote it — the same pass, primitive and grain.
  Neither
  tier holds a table: the graph owns the substrate, the local arrangement
  states `L` through it, the plane arrangement reads `L` and states `P`,
  and `arrangement_graph::identities()` is the read surface. ONE
  RELATION, TWO CLOSED LAYERS: `composed_target(source, id)` is THE ONE
  closed-rewrite read — `P∘L`, the local layer first and the plane layer
  on what it left, `-1` for a key neither retires; the carve names every
  corner through it, and an answer past the currency the carve writes in
  refuses the carve (`reseat_refusal::name_foreign`); `local_target` / `plane_target` answer one layer alone for
  that layer's own passes. `L`'s writers (the gate, the census, the
  carve) target originals and the local tier's created identities only;
  `P`'s rows may target anything; a weld's target is live and one key
  holds one target (`assert_welds_are_lawful`); `P` is closed over `L`
  for every key a plane product can hold, so a plane reader reads `P`
  alone (`assert_plane_layer_is_closed`). ONE PLANE SPACE:
  `plane_name(id)` answers the local tier's name below its extent and the
  promotion's past it, `plane_names()` every plane in plane order, so
  every reader seating a face sees every plane either tier holds. Every
  created identity is live, retired or vanished, a death being what
  nothing references at either tier —
  `account_created_identities` states each identity's account, and
  `assert_created_identities_are_accounted` checks it on a complete plane
  product under `tf::core::verification`.
  THE ONE POSITION LAW (`tf::intersect::graph::final_identity_point`): a
  key speaks what its layer's chase states, except a key retired into an
  identity whose statement encloses the read, which speaks its
  pre-mutation source; each tier evaluates at its own layer — the local
  arrangement's `point_of` through `L`, the plane arrangement through
  `P` — and the exposure composes them (`composed_identity_point`, at
  `P∘L`), so one identity has one answer per layer and no second law.
  THE RENUMBERING LAW (`arrangement/identity/extent_shift.hpp`): a plane
  value numbers created identities, planes and faces as `[world extent |
  promoted suffix]`, and a diff's world appends at its own end of each,
  so between a prior value and the next the suffix slides by the append
  count — one monotone renumbering, `extent_shift` per space
  (`forward`, `inverse`, `gap`), the three in `diff_extents`, produced
  once by `make_diff_extents`. The gap numbers below every forwarded
  suffix id, so a round's own statements outrank the standing world at
  their coincidences. A diff carries `P` across the shift
  (`carry_plane`, over `carry_plane_identities`) and composes every row
  the statement moved in `L` into `P`'s evidence (`compose_local_moves`):
  forward, then append.
- **`tf::arrangement::plane_world<Policy>`**
  (`arrangement/planes/plane_world.hpp`) —
  the ONE carrier the plane arrangement's seam speaks: the tables and
  their canonical groups, the carrier space (`n_planes`, `n_faces`,
  `frame`, `member_count`, `member`, `plane_of_face`, `descriptor`,
  `descriptors`, `face_orientation`, and at the entrance grain
  `plane_name` — answered from the substrate — and `source_side`), the
  identity space
  (`vertex_offsets`, `face_offsets`, `n_created_points`, `point_of`,
  `intersection_points`), the split statement a promoted side inherits
  (`merges` — the local layer `L` — `split_roots`, `split_survivors`)
  and the `graph`. The
  production policy (`tf::arrangement::plane_local_arrangement_policy`,
  built by `tf::arrangement::make_plane_world`) BORROWS a
  `local_arrangement` and the substrate's local tables;
  composition is compile time. ONE POLICY STATES BOTH MODES: a stock
  build carries an EMPTY promoted extension, so its suffix branch is
  never taken, and the promotion is a NEW VALUE of the same type built
  by a second constructor — the extents are frozen scalars, so a reader
  still holding the base value keeps the base extents by construction
  and no phase order can make an extent mean two things.
- **`tf::arrangement::plane_arrangement`**
  (`arrangement/planes/plane_arrangement.hpp`) — one triangulation per plane
  carrier of that prepared world. The plane's edge block IS its
  constraint set, so a preserve-mode CDT that does not refuse ends the
  plane's work; a refusal is rebuilt in resolve mode, its crossings and
  landings close into identities and splits, and the wave repeats until
  nothing new is stated (`failed()` publishes the planes still
  refusing). THE WAVE ENTRANCE obeys the same law: a round whose
  world-tier split lands on an original side, or whose closure retires an
  original, is handed back unconsumed, the reached faces enter through
  the one discovery with this tier's answered mask, and the evidence is
  seen again against the promoted tables. A weld is an identity substitution this tier owns: a plane
  whose rows only changed identity keeps the triangulation it has — one
  that refused holds none, so a substitution reaching its rows puts it
  back in the wave. It owns no identity table: it reads `L` through the
  world and writes `P`. A face this tier promotes — the refinement's
  promotion and the wave entrance alike — is seated by name through
  `seat_plane_promoted_faces`, on a standing plane its name holds or on
  one the promotion founds, so `promoted_plane_of()` names its plane; a
  founded plane `n_base_planes() + j` is framed by `promoted_frames()[j]`
  and named by the plane layer's `plane_names[j]`. `build_diff` is the
  census round: the prior value carried onto the restated world across
  the renumbering and rewaved from the statements' raw evidence. Product:
  `triangles()` (corners are FLAT identities), `slot_parents()` (per
  slot the canonical piece, `-1` for a filler diagonal),
  `corner_subs()`, the coplanar stacks (`coplanar_of`,
  `coplanar_descriptors`, `stacked`), `promoted_descriptors()` with
  `promoted_plane_of()` and `promoted_frames()`, and —
  when `record_triangle_cells()` was asked before the build —
  `triangle_cells()`. `triangulation_type::refined_cdt` runs the same
  machinery through `build_refined`.
- **Cells, incidence, fences** (`arrangement/planes/`) —
  `tf::arrangement::make_plane_arrangement_cells` numbers the recorded
  2-cells densely across the planes;
  `tf::arrangement::make_plane_piece_incidence`
  states the piece <-> cell incidence from both sides. Both identity
  spaces are dense, so counts plus one prefix build them. A cell is
  bounded by constraints and nothing else, and a cell's boundary is
  pieces and nothing else, so the incidence IS the adjacency a
  component flood walks. THE FENCE LAW
  (`tf::arrangement::make_plane_piece_fences`, the one producer of both
  verdicts): a piece carries a FAN iff an instance states the seam
  (`tf::intersect::graph::plane_edge_fan_flag`) or more than two LIVE
  cell incidences meet at it. It fences without a fan when the input
  edge is non-manifold
  (`tf::intersect::graph::plane_edge_non_manifold_flag`) or the cells
  it bounds differ in depth AND in crossing — the net winding per form
  of the sheets covering them — so a fold's own layers, which cancel,
  make no border. Everything else is `crossable`.
- **`created_points()`** — the unified table on the exact integer
  lattice: the local arrangement's points first, then everything the
  plane arrangement materialized, each where the composed layer `P∘L`
  states it (`composed_identity_point`). Created vertex ids index it
  directly; identity = `{tag, id}` pairs with created ids past the last tag.
- **Exact substrate** (`exact/`):
  `tf::exact::resolve_int_type` picks the lattice int from the input
  real; `tf::exact::vertex_converter` converts/deconverts. No predicate
  carries a band — the classifiers call the exact free predicates
  (`orient3d_sign`, `orient2d_sign`, `orient3d_plane_sign`) directly.
  Coordinates in the pipeline are lattice ints; float conversion happens
  at the edges.
  `tf::exact::input_lattice` is the operands' lattice view, built ONCE at
  the factory (`dispatch::make_graph`) over the union of the operands: the
  shared converter, the flat vertex space, and — under a tolerance — the
  sparse table of the vertices the door (`exact/door/`) moved. The band
  quantizes every exact plane (`exact/door/pool/pool_input_planes.hpp`):
  its direction onto a fixed chart whose pitch keeps two planes of one cell
  within a quarter band over the whole lattice, its intercept onto the band;
  equal keys are one pool across all forms, a key held by one plane pools
  nothing, and the pool's plane is the cell's integer direction, dense on
  the lattice, through the cell's centre offset. The rounding is the whole
  relation, so pooling is transitive and never walks from a face to its
  neighbour. Every vertex of a pooled face stands on the pool's plane
  exactly; a vertex that cannot stand on every pool around it stays put and
  every pool at it releases its faces there, to a fixed point
  (`commit_target_planes`). It lands every RIM vertex —
  on an edge one face holds — within it of another form's face exactly on
  that face's nearest feature: the corner's point, the lattice point
  nearest its projection onto an edge plus a vertex-on-edge name
  (`edge_names`, stated by `state_rim_edge_names`), or a lattice point on
  the face's committed plane (`find_rim_targets`, `commit_target_planes`,
  `place_door_vertices`). A meeting the input states is kept, interior
  vertices never move, and a vertex nothing pulled is read as at zero. At
  tolerance zero no table exists and nothing of the door executes. The
  door gives positions and edge names only; identity is the gate's; its
  refusals are counted in `census()`.
  `tf::exact::input_lattice_reader` is the ONE reader of an original's
  position (its moved row when the band moved it, the converter
  otherwise), and every tier reads originals through it.
- **`tf::csg_graph`** (`csg/csg_graph.hpp`) = an arrangement_graph plus
  the classification tier (machinery in `csg/graph/`):
  `tf::csg::graph::triangle_component_labels` (cut CCL over the cells,
  crossing a piece only where the fence allows; surface CCL over the
  uncut faces through the prebuilt `manifold_edge_link`; bridged across
  the source mesh's manifold edge), `tf::csg::graph::make_plane_radial_fans`
  (`csg/graph/make_plane_radial_fans.hpp` — one fan per fan piece, its
  pages radially ordered, admitted by the fence's `fan` verdict),
  `tf::csg::graph::make_arrangement_descriptor`,
  `make_domain_inclusions`, `compute_arrangement_domain_volumes`,
  `seed_inclusion_bits` / `propagate_inclusion_bits` (volume columns),
  and `make_sheet_sides` / `mark_sheet_wraps` / `propagate_sheet_sides`
  (sheet columns, the law in `sheet_sides.hpp`). Consumers: `make_csg_mesh` (boolean
  expressions), `make_csg_domains`, `make_outer_shell`,
  `make_intersection_curves(csg_graph)`. The build runs to the census's
  fixed point (`tf::csg::census_graph_over`): while
  `tf::csg::graph::make_census_conviction` convicts the graph, the fusions
  it states enter the standing graph as a diff, never a rebuild —
  `local_arrangement::restate_census`, then
  `tf::arrangement_graph::state_census_diff`, whose welds are the wave's
  evidence on exactly the planes the statement touched. `census()` records
  `{rounds, stated}`; `{0, 0}` is the green path.
- **`iso/`** (`tf::iso`, machinery in `iso/cut/`) — the scalar-field
  pipeline, driven by field crossings rather than by polygon
  intersections, reached only by `tf::embedded_isocurves` and
  `tf::make_isobands`, and separate from the one above. Its own umbrella
  is `trueform/iso.hpp`; `tf::make_isocontours` and
  `tf::scalar_field_intersections` are exported there too.
  `build_iso_cuts` composes it:
  `tf::scalar_field_intersections` (the sole producer of field-point
  identity, so one created point on a shared edge is one id in both
  incident faces), a vertex category per scalar, then
  `make_surface_scalar_labels` on the uncut faces beside `cut_iso_faces`
  on the cut ones, and `tf::arrangement::make_partition_ids` over both.
  The cut is the `arrangement/planes` shape with the FACE as the
  carrier: a face's chords
  are level sets of the field on it and never leave it, so
  `prepare_iso_face_cut` states its boundary chain in flat identities
  (`created ? n_original + id : id`) — the chain names each identity
  once, so it IS the point table, its position IS the local index and
  its edges are consecutive positions — plus the interior chords and the
  chain's own winding. The face then splits BY STATE. A field crosses a
  face once, so the average face carries exactly ONE chord whose ends are
  two chain positions: `split_iso_face_chord` takes the near run and the
  far run of the chain and fans each, and a triangle face's chain is
  convex by construction, which is what makes the fan the whole
  triangulation. Every other arity, a degenerate projection and any other
  chord count decline to one `tf::constrained_delaunay_triangulator`
  build in `cdt_region_mode::components`, which states the pieces AND
  their triangles; `emit_iso_face_regions` then names the corners back
  (the smaller flat id wins an output, so two faces resolve a lattice
  coincidence alike) and drops region 0 — the hull exterior and every
  non-convex pocket. `iso_band_of_triangles` is the one producer of a
  piece's band for both states.
  Product: `iso_cut_regions` (`triangles` blocked per region, `faces`,
  `minted_points`) plus the band buffer the partition reads. Both entry
  points read that one product through `gather_iso_band_triangles`, which
  assembles the band-major stream by counts and one prefix into disjoint
  ranges; the corners live in `[originals | field points | minted]`.
