# Trueform C++ Execution Patterns

This is the practical design guide for performance-critical Trueform code. It
does not inventory modules. It describes the recurring phase shapes shared by
intersection, cutting, topology, spatial search, CSG, reindexing, geometry, and
remeshing.

For each problem, identify the carrier and choose the phase shape before
choosing a container or parallel primitive.

## 1. Owning buffers and semantic ranges

Trueform separates ownership from interpretation.

| Shape | Owning form | Non-owning form |
|---|---|---|
| Linear | `tf::buffer<T>` | `tf::range` |
| Fixed blocks | `tf::blocked_buffer<T, N>` | `tf::make_blocked_range<N>` |
| Jagged blocks | `tf::offset_block_buffer<I, T>` | `tf::make_offset_block_range` |
| Points | `tf::points_buffer<T, D>` | `tf::points` |
| Polygons | `tf::polygons_buffer<...>` | `tf::polygons` |
| Segments | `tf::segments_buffer<...>` | `tf::segments` |
| Identity map | `tf::index_map_buffer<I>` | `tf::index_map` |

Buffers allocate and own flat memory. Ranges supply grouping, dereferencing,
geometry semantics, or policies without copying the underlying data. Algorithms
should accept ranges and produce owning buffers only when the result must
outlive its inputs or become a new authority.

Compose views before materialization:

```cpp
auto selected_faces = tf::make_indirect_range(face_ids, polygons.faces());
auto remapped_faces = tf::make_block_indirect_range(selected_faces, point_map);
tf::parallel_copy_blocked(remapped_faces, output.faces());
```

Relevant primitives:

- `include/trueform/core/range.hpp`
- `include/trueform/core/blocked_buffer.hpp`
- `include/trueform/core/offset_block_buffer.hpp`
- `include/trueform/core/views/offset_block_range.hpp`
- `include/trueform/core/views/mapped_range.hpp`
- `include/trueform/core/views/indirect_range.hpp`
- `include/trueform/core/views/block_indirect_range.hpp`
- `include/trueform/core/views/zip.hpp`

Do not erase known fixed arity into dynamic storage. Static block size
propagates through ranges and enables specialized iteration and copying.

## 2. Manufacture the expensive grain

Many problems begin as cheap independent observations but become expensive only
after related observations are known. Do not build a concurrent per-key data
structure. Manufacture contiguous groups:

```text
parallel unordered generation
-> parallel sort by the structural key
-> compute offsets where the key changes
-> offset-block range of related records
-> parallel expensive work over the blocks
```

Typical keys are face ID, edge identity, component label, canonical endpoint
pair, tag, region, or cross-face invariant.

Canonical skeleton:

```cpp
tf::buffer<record_t> records;
tf::generic_generate(input, records, local_state_t{}, emit_records);

tbb::parallel_sort(records.begin(), records.end(), record_less{});

tf::buffer<Index> offsets;
tf::compute_offsets(records, std::back_inserter(offsets), Index(0),
                    same_group{});

auto groups = tf::make_offset_block_range(offsets, records);
tf::parallel_for_each(groups, process_group, group_scratch_t{});
```

The sort is not cleanup. It creates the work partition, canonical ownership,
and query layout. `compute_offsets` materializes equivalence-by-adjacency into
random-addressable blocks.

Primary exemplars:

- Intersection graph crossing grouping:
  `include/trueform/intersect/graph/crossing_detection.hpp`
- Canonical edge instances and dense back-map:
  `include/trueform/intersect/graph/canonicalize_edges.hpp`
- CSG domain grouping:
  `include/trueform/csg/graph/make_csg_domains.hpp`
- Domain/component splitting:
  `include/trueform/reindex/split_into_domains.hpp`
- Topology sidedness votes:
  `include/trueform/topology/sidedness/aggregate_votes.hpp`

## 3. Offset blocks as executable structure

An offset-block structure provides more than compact jagged storage:

- block `i` is O(1) random-addressable;
- its payload is contiguous;
- blocks are independent parallel work units;
- its position can be the identity shared with another carrier;
- offsets can be rebased or copied more cheaply than rebuilding associations.

This is why offsets recur through the cutting pipeline. Intersection records for
a carrier become graph loops/edges for that carrier; those become face regions;
those become region triangulations. Payload cardinality changes at every stage,
but positional identity remains available without a hash join.

Treat offset arrays as first-class structural output. Preserve them, transform
them, or build aligned offsets deliberately.

Primary exemplars:

- `include/trueform/intersect/graph/intersection_graph.hpp`
- `include/trueform/cut/face_regions.hpp`
- `include/trueform/cut/impl/region_triangulator.hpp`
- `include/trueform/topology/structures/compute_face_membership.hpp`

## 4. Aligned jagged transformations

Use an index-preserving jagged transform when input block `i` must produce
output block `i`:

```text
A[0] A[1] ... A[n]
  |    |         |
  v    v         v
B[0] B[1] ... B[n]
```

Each output block may contain zero, one, or many elements. The output offset
array preserves the carrier association.

### `tf::generate_offset_blocks`

Use when each input element or block produces exactly one variable-length
output block. Work happens in parallel over sequential chunks; aggregation
appends payload and constructs offsets in input order.

```cpp
tf::offset_block_buffer<Index, value_t> output;
tf::generate_offset_blocks(input, output,
                           [&](const auto &element, auto &local_data) {
                             emit_block(element, local_data);
                           });
```

Implementation: `include/trueform/core/algorithm/generate_offset_blocks.hpp`.

### `tf::blocked_reduce_sequenced_aggregate`

Use when each input block produces several correlated outputs, offsets require
custom rebasing, or aggregation must update multiple carriers together.

```cpp
struct local_t {
  tf::buffer<Index> sizes;
  tf::buffer<value_t> values;
  tf::buffer<descriptor_t> descriptors;
  scratch_t scratch;
};

tf::blocked_reduce_sequenced_aggregate(
    input, global, local_t{},
    [](auto block, local_t &local) {
      for (const auto &element : block)
        process_serially(element, local);
    },
    [](const local_t &local, global_t &global) {
      append_and_rebase_in_input_order(local, global);
    });
```

The sequencer is not merely a determinism feature. Input-ordered aggregation can:

- preserve `input[i] -> output block[i]`;
- construct CSR offsets during append;
- keep several jagged carriers positionally aligned;
- rebase block-local IDs with one running base;
- retain grouping created by an earlier sort;
- avoid a later sort, lookup, or permutation map.

Use ordinary `blocked_reduce` when aggregation order has no structural value or
the next phase canonical-sorts the records anyway.

Implementation:
`include/trueform/core/algorithm/block_reduce_sequenced_aggregate.hpp`.

Primary exemplars:

- Intersection graph loops and edges:
  `include/trueform/intersect/graph/intersection_graph.hpp`
- Face-region loops, holes, and descriptors:
  `include/trueform/cut/face_regions.hpp`
- Region triangulation and recovery output:
  `include/trueform/cut/impl/region_triangulator.hpp`
- Per-edge face links:
  `include/trueform/topology/structures/compute_face_link_per_edge.hpp`

## 5. Count, prefix, allocate, materialize

Unknown total output size does not imply concurrent append.

```text
parallel per-carrier counts or byte sizes
-> one prefix scan
-> exact global allocation
-> parallel writes into disjoint slices
```

Use this when a count pass is cheaper than building and merging local payloads,
or when the output can be written directly once its range is known.

Benefits:

- one allocation;
- no locks or atomics;
- no repeated growth;
- no rebasing after append;
- stable correspondence between counts, offsets, and payload.

Primary exemplars:

- CDT face extraction:
  `include/trueform/topology/constrained_delaunay_triangulator.hpp`
- Face membership:
  `include/trueform/topology/structures/compute_face_membership.hpp`
- Tube batches: `include/trueform/geometry/make_tube_mesh.hpp`
- OBJ output: `include/trueform/io/write_obj.hpp`
- Dynamic CSG output: `include/trueform/csg/graph/make_csg_mesh.hpp`

## 6. Choose parallel state by work shape

### Partitionable input

The normal model is state carried by the partitioning primitive and reused by a
tight serial kernel within that block.

| Need | Primitive |
|---|---|
| Independent side effects plus scratch | `parallel_for_each(r, f, State{})` |
| Unordered variable output | `generic_generate` |
| Ordered variable output | `sequenced_generate` |
| One jagged output block per input | `generate_offset_blocks` |
| Custom unordered aggregation | `blocked_reduce` |
| Custom positional aggregation | `blocked_reduce_sequenced_aggregate` |

Local state is per task/block, not one persistent object per arena worker. It
may contain reusable buffers and scratch when copying/constructing it once per
block is appropriate. Clear logical contents and retain capacity across the
serial inner loop.

### Irregular callback traversal

Use `local_value`, `local_buffer`, or `local_vector` only for irregular parallel
tree traversal whose task-driven callback cannot expose a stable partition or
receive block state.

Examples:

- `include/trueform/intersect/polygon_intersections.hpp`
- `include/trueform/intersect/impl/face_pair_search.hpp`
- `include/trueform/spatial/tree/local_tree_metric_result.hpp`

Do not use arena-local storage merely because output is variable-length.

## 7. Dense maps and addressability

When IDs occupy `[0, n)`, direct indexing is the default.

### Sentinel map

Allocate `n` entries, fill with an out-of-range sentinel, and use O(1) lookup.
This is appropriate for old-to-new identity, membership, created-point lookup,
and sparse facts over a bounded domain.

### Generation or watermark map

When many groups reuse the same global keyspace, avoid clearing `n` entries per
group. Store the current generation or use a monotonically advancing value
range so stale entries are distinguishable.

Examples:

- `include/trueform/reindex/split_into_components.hpp`
- `include/trueform/reindex/split_into_domains.hpp`
- `include/trueform/csg/graph/make_csg_domains.hpp`

### Sparse sorted table

For genuinely rare global facts, sort by key and use `lower_bound` or
`equal_range`. Consolidate transitive or duplicate facts once so lookups do not
chain.

Examples:

- Region-triangulator split and merge tables:
  `include/trueform/cut/impl/region_triangulator.hpp`
- Sparse loop-to-hole index:
  `include/trueform/cut/face_regions.hpp`

### Hash maps

Hash maps are almost never the answer. Before using one, rule out:

- flattening local/tagged identities into a dense global space;
- sorting and grouping records;
- a dense sentinel/generation map;
- a sorted sparse table;
- a tiny linear or stack container.

Only use a hash map when the active key set is genuinely sparse/dynamic or the
key is arbitrary/composite, direct addressing would require clearing or
allocating a disproportionate global domain, and an existing Trueform exemplar
has the same work shape. Sparse per-query traversal sets such as k-rings and
neighborhood searches are the relevant exception; bounded reusable identity
tables are not.

## 8. Sort, compact, and retain the back-map

Sort compact IDs or records and derive structure from adjacency. When physical
order changes, retain a dense map so every old reference remains random-
addressable.

Useful primitives:

- `tf::make_unique_index_map`
- `tf::make_unique_and_index_map`
- `tf::mask_to_index_map`
- `tf::ids_to_index_map`
- `tf::compose_index_maps`
- `tf::parallel_copy_by_map_with_nones`

An index map normally carries both directions:

- dense old ID -> new ID or sentinel;
- compact new ID -> kept old ID.

Compose maps across phases rather than applying and rediscovering intermediate
geometry.

Primary exemplars:

- `include/trueform/core/algorithm/make_unique_index_map.hpp`
- `include/trueform/topology/half_edges.hpp`
- `include/trueform/intersect/graph/canonicalize_edges.hpp`
- `include/trueform/remesh/isotropic_remesh.hpp`

## 9. Parallel over-segmentation and equivalence collapse

Some irregular traversals cannot maintain final components cheaply while they
run. Permit provisional identities:

```text
parallel walkers claim work with provisional labels
-> collisions emit pairs of equivalent labels
-> barrier
-> union-find collapses provisional labels
-> dense compact label map
-> parallel substitution into final labels
```

This keeps the expensive traversal parallel and reduces shared coordination to
small claims/counters. Workers record evidence; they do not maintain the final
partition concurrently.

The dense and sparse equivalence-class builders are union-find followed by
compact class assignment:

- `tf::make_dense_equivalence_class_map`
- `tf::make_sparse_equivalence_class_map`

Implementation:
`include/trueform/core/algorithm/make_equivalence_class_map.hpp`.

Primary exemplar:
`include/trueform/topology/components/finder.hpp`.

The finder atomically claims vertices, records only provisional-label
collisions, collapses those labels after the walkers join, and applies the dense
label map in parallel. Its local collision set and arena-local output are a
narrow irregular-traversal exception, not the default state model.

## 10. Discovery, materialization, consumption

Never let independent workers assign shared global identity by timing.

```text
DISCOVERY
  parallel workers emit canonical proposals or facts

MATERIALIZATION
  sort/deduplicate/fuse
  assign offsets and IDs exactly once
  update the authoritative table

CONSUMPTION
  parallel workers read the frozen authority
```

When more facts may be discovered after consumption, repeat this as a bounded
wave and explicitly track the dirty carrier.

Primary exemplars:

- Intersection finalization:
  `include/trueform/intersect/polygon_intersections.hpp`
- Intersection graph crossing splits:
  `include/trueform/intersect/graph/intersection_graph.hpp`
- Region-triangulator refinement and recovery:
  `include/trueform/cut/impl/region_triangulator.hpp`
- Remesh edge splitting:
  `include/trueform/remesh/split/half_edge_splitter.hpp`

## 11. Transform proven output

If an upstream phase already proved or computed a fact, downstream changes
should transform that output:

- substitute merged identities;
- compact collapsed elements;
- compose index maps;
- remap clean blocks;
- rebuild only marked dirty slots;
- derive several consumers from one expensive query record.

Do not rerun an exact intersection, triangulation, ray cast, or topology build
merely because IDs were compacted or a global equivalence was learned.

Primary exemplars:

- Stitched connectivity:
  `include/trueform/topology/stitched_face_membership.hpp`
- Stitched manifold links:
  `include/trueform/topology/stitched_manifold_edge_link.hpp`
- Region weld conformance:
  `include/trueform/cut/impl/region_triangulator.hpp`
- CSG inclusion and nesting evidence:
  `include/trueform/csg/graph/seed_inclusion_bits.hpp`

## 12. Parallel mutation through separation

Do not make shared topology mutation thread-safe with locks. Create independent
mutation domains:

```text
partition topology
-> freeze separators/frontiers
-> mutate independent interiors concurrently
-> join
-> repair or process the residual frontier
-> rebuild derived counters/maps
```

Primary exemplar:
`include/trueform/remesh/collapse/collapse_to_exhaustion_parallel.hpp`.

Within each partition, scoring or discovery may be parallel while the actual
dependency chain remains serial.

## 13. Exact identity and predicate specialization

Exact vertices carry coordinate and identity as separate facts. SoS ordering is
defined by identity. Shared split points are carried as parameters on an
authoritative edge and materialized consistently by each carrier.

Do not assume one generic robust-predicate strategy. Trueform chooses direct
wide-integer evaluation, bounded exact arithmetic, or a certified floating
filter based on operand width, reuse, and measurement. Every path must preserve
the exact sign.

Primary exemplars:

- `include/trueform/exact/orient2d.hpp`
- `include/trueform/exact/orient3d.hpp`
- `include/trueform/exact/incircle.hpp`
- `include/trueform/exact/segment_intersect.hpp`
- `include/trueform/topology/cdt_refiner.hpp`

## 14. Review checklist

Before accepting a hot-path design, verify:

- The carrier and independent grain are explicit.
- Flat ownership and view composition are preserved.
- Variable output uses unordered generation, ordered generation, or a count pass
  deliberately.
- `blocked_reduce_sequenced_aggregate` is used only when order creates or
  preserves structure, and is used whenever positional jagged alignment needs
  it.
- Global IDs are assigned after discovery, not by worker timing.
- Dense bounded identities are directly addressable.
- Hash maps, TLS, atomics, and locks have a work-shape justification.
- Producer-owned facts are not rederived.
- Clean output is transformed rather than recomputed.
- Serial phases are small, structural, dependent, or below a measured threshold.
- Correctness is proven before performance is measured.
