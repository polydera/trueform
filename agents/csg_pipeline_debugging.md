# CSG Pipeline Debugging Strategy

> **Task-specific reference.** Read this when debugging a CSG correctness
> failure. This is a tracing method, not a description of every CSG algorithm.

## The central rule

Do not start from bad output geometry and try to reconstruct which input points,
intersection points, or faces produced it.

The output is the last materialization boundary. Before it reaches that boundary,
points may have been created, sorted, remapped, welded, substituted, compacted,
or duplicated as topological occurrences. Equal coordinates do not recover that
history.

Debug forward from the producer-owned records and carry identity through the
maps the pipeline already exposes:

```text
intersection records
-> intersection graph
-> face regions
-> region triangulations
-> component labels and chosen sides
-> output map data
-> raw output
```

At every arrow, ask only:

> Did the same topological fact survive this carrier change correctly?

Stop at the first transition where it did not.

## Start with intersection records

Print the complete records for the smallest failing face pair.

The stable identity is the tagged primitive ownership:

```text
(tag, object, target, tag_other, object_other, target_other)
```

The point ID names a point-pool entry. It is useful inside one run, but it is not
a substitute for the record's topological ownership and may not be stable across
runs.

Check:

- both directions of the face pair report the same intersection facts;
- vertex records name the correct local endpoint on every duplicated edge;
- explicit flags, not record count or coordinates, decide special cases such as
  coplanarity.

If the records are wrong, stop there. Do not debug the graph or triangulator.

## Follow explicit remaps

Every later stage has an explicit relationship to its producer. Use it.

- Account for the intersection graph's point remap.
- Inspect the exact graph loop or edge carrying the record identities.
- Use the face-region descriptor and loop index to find the corresponding
  region.
- Use the same region index to inspect its triangulation range.
- Use exposed ranges to follow a region label to emitted triangles.
- Use the partition and CSG map data to follow selected identities into the raw
  output.

Never match stages by nearest coordinates. Never assume equal coordinates mean
equal topology.

Created numeric IDs can vary between runs. Either keep the entire trace in one
run or key the trace by stable record ownership.

## Regions own arrangement topology

`face_regions` is the authoritative topology used downstream.

`region_triangulator` only materializes each region. Its diagonals, recovery
subedges, and coincident triangle edges are not new arrangement adjacency.

Therefore always distinguish:

1. a defect already present in a region loop or its connectivity;
2. a correct region whose triangulation is wrong;
3. correct regions and triangles whose final point mapping incorrectly unifies
   distinct topological occurrences.

Two emitted triangles sharing the same endpoint IDs does not prove their source
regions are duplicates. Triangulation can make unrelated region arcs look like
the same mesh edge. Do not feed that apparent triangle topology back into CSG
classification.

## Debug raw output through provenance

When the raw mesh is open or non-manifold:

1. Print the exact bad edge IDs and all incident output faces.
2. Use returned source tags and source faces to identify the selected producers.
3. Use returned map data to translate output point IDs back to original or
   created identities.
4. Inspect those identities in the corresponding regions and triangulation
   ranges.

Do not locate producers by searching for equal output coordinates.

Run this check before `cleaned()`. Cleanup may hide the defect by welding points
or removing duplicate faces; that is not proof that the CSG pipeline was
correct.

## Openness triage: read the NM-edge fans before anything else

An open boolean result has exactly three possible authors: the arrangement,
the triangulation, or the classification. Separate them before investigating
anything, using the structure classification itself consumes —
`csg_graph::descriptor().fans` (`tf::cut::non_manifold_edge_fans`): the
non-manifold edges with their incident region fans.

Five readings, in this order:

0. **Per-face region partition.** Group regions by their source face and
   count every walk edge. Each interior cut edge must appear exactly twice
   (two sibling regions), and the edges appearing once must form a single
   closed cycle — every vertex degree 2, no odd degree, no pinch. This is
   threshold-free and it is the test that says whether the region
   decomposition itself is sound. Do **not** substitute a total-area
   comparison against the original flat face: quantization snaps an
   intersection point off the face's edge line, so the regions legitimately
   tile a slightly deformed triangle while staying watertight against the
   neighbour sharing that same snapped point. The area form needs an
   arbitrary threshold and will report defects that are not there.

1. **Boundary conservation.** For every live region, every edge of its walks
   (boundary walk plus hole walks) must appear as an edge of that region's
   triangulation range. A region that loses a walk edge tears a hole against
   whatever region shares that edge. This is the cheapest reading, it
   localises to a named region immediately, and it settles the arrangement /
   triangulation split on its own — run it first. A triangulator that
   resolves crossings by rerouting a constrained edge through another vertex
   violates it, and the violated edges are exactly the result's boundary
   edges.
2. **Open fragments.** `labels().open_component_mask()` must be all zero for
   closed operands. Self-merge is a fragment property; intersecting closed
   meshes cannot produce an open fragment, so a set bit is an arrangement
   defect on its own.
3. **Per-fan-region accounting.** For each boundary edge that *is* a fan
   edge, print every region in its fan with three facts: does its walk carry
   the edge, how many of its triangles carry the edge, and its component and
   chosen side. Exactly two incidences should be selected and each selected
   region should emit exactly one triangle on the edge. `in_walk=1` with
   `tris_on_edge=0` is a triangulation fault; `in_walk=0` means the
   connectivity fabricated the fan and the merges are wrong.
4. **Fan parity.** Every form contributes an even number of incidences to a
   non-manifold edge, so a fan of odd size means one incidence is missing —
   *unless* a coplanar contact collapsed there. A collapsed wall is one
   surface shared by two forms and legitimately contributes once. Before
   calling an odd fan a defect, list the regions whose walk carries the edge
   and check whether the ones the fan omits are `dead_loops()` entries paired
   through `coplanar_pairs()`. Coplanar contacts also break vertex-degree
   parity along the collapsed patch's boundary, so degree parity alone never
   settles the question.

Do not read a boundary edge's absence from the fan table as evidence of a
T-junction. It only says the edge is interior to some triangulation, which a
conservation violation produces just as readily as an unbroadcast vertex
does. Distinguish them by testing exactly, on the lattice, whether a created
point actually lies on the segment.

## Order of investigation

Use this order for every failure:

1. Reproduce at tolerance `0`.
2. Verify the two inputs are closed and manifold.
3. Find the first failing operation in an iterated sequence.
4. For an open result, run the NM-fan triage above to choose the branch.
5. Print the complete relevant intersection records.
6. Follow their identities into the intersection graph.
7. Check the corresponding face regions and region connectivity.
8. Compare each region with its triangulation range.
9. Check component labels, chosen sides, and selected source regions.
10. Translate raw output IDs back through map data.
11. State the first broken transition before proposing a fix.

The proof should name exact records, identities, region IDs, and source
descriptors. Geometry is used to evaluate predicates inside the algorithms, not
to reconstruct topology during debugging.
