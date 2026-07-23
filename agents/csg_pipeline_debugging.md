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

## Order of investigation

Use this order for every failure:

1. Reproduce at tolerance `0`.
2. Verify the two inputs are closed and manifold.
3. Find the first failing operation in an iterated sequence.
4. Print the complete relevant intersection records.
5. Follow their identities into the intersection graph.
6. Check the corresponding face regions and region connectivity.
7. Compare each region with its triangulation range.
8. Check component labels, chosen sides, and selected source regions.
9. Translate raw output IDs back through map data.
10. State the first broken transition before proposing a fix.

The proof should name exact records, identities, region IDs, and source
descriptors. Geometry is used to evaluate predicates inside the algorithms, not
to reconstruct topology during debugging.
