# The Trueform Performance Philosophy

These laws explain why Trueform does not look like conventional geometry code.
They govern design choices; concrete phase shapes and primitive selection live
in `cpp_execution_patterns.md`.

## 1. Data shape is part of the algorithm

Own data once in flat buffers. Recover semantic structure through lightweight
ranges: blocked, offset-blocked, mapped, indirect, zipped, and policy-tagged.
Nested ownership, pointer graphs, and intermediate geometry usually destroy the
locality and composability the algorithm needs.

Preserve static arity in the type and storage. Use offsets only for genuinely
jagged data. Materialize only when an ownership boundary requires it.

Allocation lifetime is part of data shape. Exact refactors preserve allocation
points, retained capacity, scratch reuse, ownership, and release timing unless
a deliberate measured change proves another shape better.

## 2. Manufacture the expensive grain

The useful independent work unit often does not exist in the input. Generate
cheap records without ordering constraints, parallel-sort them by the key that
defines shared work, compute offsets at key boundaries, then process the
resulting contiguous groups independently.

The extra passes are the fast path. They replace synchronized insertion,
associative lookup, scattered reads, and arbitrary task graphs with parallel
flat passes and tight serial kernels.

## 3. Preserve carrier identity through jagged stages

Offset blocks are executable structure, not just variable-length storage. Block
index `i` can identify the same face, loop, region, edge, or component across a
pipeline even when every stage produces a different number of elements.

Use `generate_offset_blocks` or a `blocked_reduce_sequenced_aggregate` whose
aggregator constructs offsets when positional correspondence is part of the
result. `sequenced_generate` preserves flat input order, but without offsets it
does not preserve one jagged block per input. Ordered aggregation is valuable
because it can preserve an implicit join, construct CSR offsets directly,
enable rebasing, and remove later sorting or lookup.

## 4. The map is latent in the data

Identity usually already exists as a canonical coordinate or structural key: a
tagged flat ID, an endpoint pair, a dyadic parameter, or a sorted record key.
Expose it instead of adding bookkeeping.

The common shape is:

```text
parallel generate
-> parallel sort
-> one adjacency/group sweep
-> dense map or offsets
-> parallel apply
```

For repeated local work over a bounded global ID space, reuse one dense array
with a generation or watermark. Hash maps are rare exceptions, not a peer
representation choice.

## 5. Every fact has one authority

The producer records the facts it proves: exact identity, coplanarity, ancestry,
split parameters, adjacency, merge evidence, and classification inputs.
Consumers read the authoritative carrier. They do not rederive the same fact
from coordinates or rebuild a structure that already exists upstream.

Carry identities and parameters through the pipeline; materialize geometry once
where the authority says it belongs.

## 6. Transform outputs; do not rerun producers

A changed global fact such as a merge, weld, compaction, or reindexing is usually
a substitution over existing output plus removal of what collapsed. Preserve
clean ranges, remap references, and recompute only the explicitly dirty
frontier.

Recomputation is slower and may produce a different identity history.

## 7. Work at the grain where the question lives

Parallelize the largest independent carrier. Inside it, keep dependent work
serial, cache-local, and stateful. A face may be parallel while its edges are a
tight loop; tree tasks may be parallel while leaf Cartesian products are serial;
partitions may mutate concurrently while each topology walk remains serial.

When a design changes carrier, enumerate what the old carrier guaranteed by
construction. Those invariants do not move automatically.

## 8. Partition-carried state is the default

For partitionable input, state belongs to the sequential block created by the
parallel primitive. Reuse it across that block with `parallel_for_each` state,
`generic_generate`, `generate_offset_blocks`, or a blocked reduction.

Arena-indexed `local_value`, `local_buffer`, and `local_vector` storage is
reserved for irregular parallel tree traversal whose callback/task API has no
stable partition or state slot. It is not a general variable-output or scratch
mechanism.

## 9. Correctness comes from phase structure

Workers propose; one phase materializes shared identity; later phases consume
the frozen result. Parallel walkers may create provisional components and emit
only collision pairs; union-find collapses them after the barrier. Recovery
runs as bounded discovery/materialization/retry waves.

Remove hazards by shape rather than surrounding a mutable structure with
defensive synchronization.

## 10. Topology by identity; geometry by controlled authority

Coordinate equality is not topological identity. Exact vertices carry both.
SoS ordering uses identity, not accidental coordinate order. Shared split
parameters let separate carriers materialize the same point consistently.

Where coordinates must decide, keep the site singular, exact or certified, and
make it report the resulting global identity fact.

## 11. Serial phases are part of parallel algorithms

Prefixes, adjacent sweeps, union-find, offset rebasing, small sorts, graph walks,
bucket queues, leaf kernels, and incremental triangulation can be the fastest
correct implementation. "Parallel by default" means the correct outer carrier,
not every loop.

Use sequenced aggregation only when order creates structure. If the next phase
canonical-sorts the records anyway, unordered aggregation is cheaper and valid.

## 12. Nothing is believed until measured

First prove identical output with an oracle capable of failing under the bug
class. Then benchmark the real workload, both threadings, from fresh binaries,
using repeated runs.

Thresholds, fast paths, direct-exact versus filtered predicates, and rare
exceptions to the standard phase shapes are decided by measurement—not generic
C++ or computational-geometry folklore.

## 13. Split by state; merge solutions, not structures

When carriers divide into states with different work — cut and uncut, dirty
and clean, failed and passed — solve each state on its own structure: the
small changed state on a purpose-built structure scaled to it, the large
unchanged state on the structure that already exists, reached through a mask
and an applier. The graph stays implicit; an applier walking
`manifold_edge_link` under a mask IS the uncut connectivity, at zero build
cost.

Merge the solutions afterward, never the structures: emit bridge pairs where
the states meet and collapse the label spaces with a dense equivalence map.
Aggregate labels, not edges.

One unified structure over both states pattern-matches to thoroughness and is
the anti-pattern: it re-derives what the untouched state already knows, and
its cost scales with the whole input instead of with the change.

## 14. The average path funds nothing

Machinery that serves only the exceptional path — recovery connectivity,
split-propagation tables, sharer indexes — is built on the failing frontier
when failure is observed, never globally in advance. The CDT that just works,
the face that was never cut, must pay zero for structures they will not use.

The same law forbids re-materializing what an existing structure already
answers. The source mesh is the authority for uncut geometry; a stream copy of
it is a second producer with a memory-bandwidth bill.
