# CSG Pipeline Debugging Strategy

> **Task-specific reference.** Read this when debugging a CSG correctness
> failure. It is a tracing method over the current pipeline, not a description
> of every algorithm in it.

## The central rule

Do not start from bad output geometry and try to reconstruct which input
points, intersection points, or faces produced it.

The output is the last materialization boundary. Before it reaches that
boundary, identities were named, split, welded, substituted, promoted,
compacted, or duplicated as topological occurrences. Equal coordinates do not
recover that history, and the pipeline runs on an exact integer lattice whose
deconversion to the input real is itself a coordinate change.

Debug forward from the producer-owned surfaces, carrying identity through the
tickets the pipeline already publishes. At every arrow, ask only:

> Did the same topological fact survive this carrier change correctly?

Stop at the first arrow where it did not.

## The chain, and who owns each fact

```text
forms
  -> tf::polygon_intersections                point NAMES and contact records
  -> tf::intersect::graph::local_arrangement   planes, frames, members, the
                                               post-split definition tables,
                                               merges, entrants
  -> tf::arrangement::plane_world<Policy>     the one carrier the plane tier
                                               reads; one world states both
                                               modes and promotion is a VALUE
  -> tf::arrangement::plane_arrangement       one triangulation per plane
                                               carrier, plus the recovery wave
  -> cells -> piece incidence -> piece fences  the arrangement's own structure
  =  tf::arrangement_graph                     the exposed stream, the created
                                               points, failed()
  +  classification                            components, radial fans,
                                               descriptor, domains, sheets
  =  tf::csg_graph
  -> make_csg_mesh / make_csg_domains / make_outer_shell /
     make_intersection_curves
```

One fact, one producer. The table below is the routing rule: read the fact
where it is produced, never re-derive it downstream.

| Fact | Produced by | Read through |
|--|--|--|
| Point identity (a canonical NAME, not a coordinate) | `tf::polygon_intersections` | `n_vertex_points()`, `vertex_anchor`, `home_edge`, `exact_parameter` |
| Which faces share a geometric plane | the plane graph inside `local_arrangement` | `world().frame/member_count/member/plane_of_face/descriptors` |
| Every constraint of every plane, splits already applied | `local_arrangement` | `world().edge_defs()`, `canon_group()`, `plane_edges(plane)` |
| Retired identities | the identity gate | `world().merges()` (closed — one search) |
| The triangles, and the piece each slot belongs to | `plane_arrangement` | `arrangement().triangles()`, `slot_parents()` |
| Which carriers hold no product | `plane_arrangement` | `failed()` |
| 2-cells and their piece boundary | `make_plane_arrangement_cells` / `make_plane_piece_incidence` | `cells()`, `piece_incidence()` |
| Whether a piece fans, and whether the flood may cross it | `make_plane_piece_fences` | `piece_fences().fan` / `.crossable` |
| Components | `tf::csg::graph::triangle_component_labels` | `labels()` |
| Domains, bundles, radial fans | `make_arrangement_descriptor` | `descriptor()` |

Never match stages by nearest coordinates. Never assume equal coordinates mean
equal topology. Created numeric ids vary between runs — keep the whole trace in
one run, or key it by stable ownership.

## Step 0: the completeness gate

`tf::arrangement_graph::failed()`
(`arrangement/arrangement_graph.hpp`) and its forward
`tf::csg_graph::failed()` (`csg/csg_graph.hpp`) are the first read of every
investigation. They publish the arrangement planes that hold no product.

EMPTY IS THE COMPLETENESS CLAIM: every plane that bounds area and carries
constraints holds the triangulation of them, so every face of every operand
holds the span the graph exposes for it.

A non-empty `failed()` ends the search at the plane tier, because a failed
plane is silent downstream: its faces keep `descriptor.plane != -1`, so they
are classified cut, so the originals are excluded from the surface labels, and
the empty triangle span becomes a hole nothing else reports. Debugging
classification against a non-empty `failed()` is wasted work.

One state is a stated exclusion, not a defect: a carrier with an empty
constraint set — `world().plane_edges(plane)` empty for a world-owned plane, or
`arrangement().current_plane_def_rows(ticket)` empty for a plane the wave moved
to the local tier.

A plane whose exact frame is a LINE (`world().frame(plane).plane_n` all zero)
bounds no area, so it emits no triangle. That is its product, not an absent
one: it runs the same triangulation as every other carrier, states the
crossings and welds its constraint set holds, and appears in `failed()` on the
same terms — only when its constraint set refused every round.

Quality is a different question. `arrangement().refinement_refused_planes()`
(`arrangement/planes/plane_arrangement.hpp`) names carriers the refinement
producer declined to plan for; the stock kernel triangulates them in the same
pass, so they hold an unrefined product and correctly never appear in
`failed()`.

The fixtures that pin this surface are
`tests/arrangement/planes/test_plane_refusal_recovery.cpp` and
`tests/csg/test_csg_refusal_recovery.cpp`. Both walk a stranded scene that the
recovery wave rescues, its neighbourhood ladder, and an average scene that
never refuses. The nonzero side is stated by hand — a fixture world with one
plane's edge block emptied — because no pipeline input is known to strand a
carrier permanently; reproduce a nonzero `failed()` that way, not by hunting
for one.

## Step 1: the censuses say which stage did the work

Three counters are published and cost nothing to read. They do not prove
correctness; they say where to look and whether a suspected path ran at all.

| Census | Surface | Read it for |
|--|--|--|
| `tf::intersect::graph::plane_census` | `graph.world().census()` | `crossings_ee` / `crossings_ve` / `collinear_pairs` (what detection found), `classes`, `same_root_collapses` / `cross_root_merges` (what the identity gate absorbed), `splits_out_of_span` / `splits_on_endpoint`, `defs_in` / `defs_out` |
| `tf::arrangement::plane_arrangement_census` | `graph.arrangement().census()` | `refusals` and `rounds` (did the recovery wave run, and how far), `crossings` / `landings` / `collisions` / `splits`, `created`, `triangles` / `dead_triangles`, `failed_planes` |
| `tf::arrangement::plane_refinement_census` | `graph.arrangement().refinement_census()` | `promoted_planes`, `affected_planes`, `rings_run`, `ring_boundary_splits`, `physical_delivery_rows`, `refused_planes` |

`rounds == 1` with `refusals == 0` means the preserve-mode CDT ended every
plane's work and no recovery wave ran — a defect there is not a recovery
defect. `refused_planes` nonzero with `failed()` empty is the refinement
declining, not a hole.

Two counters record an anomaly with no consumer, and are not failures:
`plane_census::names_truncated` (a crossing name of more than three planes) and
`descriptor().n_invalid_fans` (a lone-page fan, deliberately never walked).

## Step 2: the intersection records

Print the complete records for the smallest failing face pair.

The stable identity is the tagged primitive ownership — the tuple
`tf::intersect::tagged_intersection` itself orders and compares on
(`intersect/records/tagged_intersection.hpp`):

```text
(tag, object, target, tag_other, object_other, target_other)
```

The record's `id` is a canonical point NAME, not a slot in a coordinate table.
Below `n_vertex_points()` it is an original vertex (`vertex_anchor`); at or
above it, an exact parameter class on an original edge (`home_edge` +
`exact_parameter`). No coordinate is computed or kept at this tier, so there is
nothing here to compare geometrically.

Check:

- both directions of the face pair report the same intersection facts;
- vertex records name the correct local endpoint on every duplicated edge;
- explicit flags, not record count or coordinates, decide special cases —
  `tf::intersect::coplanar_pair_flag` is the coplanarity statement, decided once
  at the pair call on exact coordinates.

If the records are wrong, stop there. Do not debug the plane tier.

## Step 3: the plane graph and the definition tables

`graph.world()` is the `local_arrangement`. Its product is the post-split
definition table, canon-major with a rebuilt plane CSR — **there is no splits
side table; the definitions ARE the splits, applied**. A "split that was not
applied" therefore shows up directly as a definition still spanning a whole
root, not as a missing entry somewhere else.

Reads: `edge_defs()`, `n_canon()`, `canon_group(canon)`, `plane_edges(plane)`,
`descriptors()`, `frame(plane)`, `member_count(plane)` / `member(plane, i)`,
`plane_of_face(face)`, `face_orientation(face)`.

A definition states everything about itself. Its flags
(`intersect/graph/plane_edge_def.hpp`) carry the input facts the fence and the
radial tier consume later:

| Flag | Means |
|--|--|
| `plane_edge_fan_flag` | the instance is an intersection edge, so planes meet here and a radial pairing stands |
| `plane_edge_non_manifold_flag` | the canonical pair carries more than two side statements — the mesh's own non-manifold edge |
| `plane_edge_reversed_flag` | key order and base-loop emission order disagree |
| `plane_edge_whole_side_flag` | the boundary instance covers its original edge whole |
| `plane_edge_radial_flag` (+`_reversed`) | the instance carries its producing pair's carrier-line orientation |

`merges()` is the closed rewrite table: one search states what a retired
identity speaks now. Consumers holding an old id are merge-blind by design — do
not add a second resolution of a merge anywhere downstream.

`entrants()` with `entrant_descriptors()`, `entrant_planes()` and
`entrant_orientations()` are the source faces the cut world never named but a
split or a weld reached. A face that appears in the arrangement without an
intersection record is normally an entrant, not a defect.

## Step 4: the plane arrangement

One triangulation per plane carrier of the prepared world. The plane's edge
block IS its constraint set, so a preserve-mode CDT that does not refuse ends
that plane's work; a refusal is rebuilt in resolve mode, its crossings and
landings close into identities and splits, and the wave repeats until nothing
new is stated.

One world states both modes. The production policy holds the base value; the
promotion is a NEW VALUE of the same type with extents frozen at one barrier,
so a reader holding the base value keeps the base extents by construction.
`n_base_faces()` / `n_base_planes()` divide the world's own carriers from what
refinement promoted; `promoted_descriptors()` states row `i` for face
`n_base_faces() + i` on plane `n_base_planes() + i`.

Instruments on the product:

- `plane_range(p)`, `plane_triangles(p)`, `face_range(face)` — the spans;
- `plane_tickets()` — `-1` means the plane still reads the world tables,
  anything else is its local block, read through `current_plane_defs(block)` /
  `current_plane_def_rows(block)`;
- `slot_parents()` — per triangle, per slot, the canonical piece ticket, `-1`
  for a filler diagonal. **A filler diagonal is not arrangement adjacency.**
  Two emitted triangles sharing endpoint ids does not make their sources
  duplicates, and that apparent triangle topology must never be fed back into
  classification;
- `piece_definitions(world, ticket)` — the complete definition span of a
  ticket, with `immutable_piece_extent()` splitting the caller-owned prefix from
  the PA-owned suffix and `final_piece_ticket_extent()` bounding the address
  space (which is NOT the live-piece count);
- `corner_subs()` — per corner, where it sits on the emitting member's own
  polygon;
- `triangle_cells()` — only when `record_triangle_cells()` was asked before the
  build; `arrangement_graph` always asks.

A weld is an identity substitution this tier owns. A plane whose rows only
changed identity KEEPS its triangulation and remaps corners through the closed
table; a plane that refused holds none, so a substitution reaching its rows
puts it back in the wave. When triaging a weld, the question is not "did the
coordinates merge" but "was the carrier re-triangulated exactly when its
constraint set changed".

The laws this tier publishes are pinned in
`tests/arrangement/planes/test_plane_arrangement_laws.cpp` and read only off
the published surface: **L** the live set is exactly the pieces the emitted slots
name, **W1** one flat endpoint key per live piece, **W2** every carrier plane of
a live piece states it, **O** a plane with ticket `-1` may name no PA-owned
suffix piece, **I** a live cut piece's emitted slot count is 2 when every
instance on that plane is an interior cut and 1 when a base-loop instance shares
it. The same file proves its instrument can fail, so a green run there is
evidence rather than a tautology.

That file also carries THE `null_refined` ORACLE, which is how a refined-path
defect is separated from a pre-existing one: a refinement configured to split
nothing must publish records identical to the stock build. Red there means the
refined path corrupted a product the stock path states correctly. Green there,
with a refined scene still wrong, means the defect is in what the refinement
legitimately added — look at `refinement_census()` and the promoted rows.

## Step 5: cells, incidence, and the fence verdicts

A cell is bounded by CONSTRAINTS and nothing else, so a walk inside one crosses
no edge the arrangement stated and a filler diagonal never leaves the cell it
was cut in. A cell's boundary is pieces and nothing else, so the incidence IS
the adjacency the component flood walks.

- `cells().cell_of[row]`, `cells().n_cells` — the 2-cells, numbered densely
  across the planes;
- `piece_incidence().rows_of_piece[piece]` — the rows naming a piece, each
  `triangle * 3 + slot`, ascending;
- `piece_incidence().pieces_of_cell[cell]` — the other direction;
- `piece_fences().fan[piece]` and `.crossable[piece]` — the verdicts.

THE FENCE LAW (`arrangement/planes/make_plane_piece_fences.hpp`, the one
producer of both verdicts). A piece carries a FAN iff an instance states the seam
(`plane_edge_fan_flag`) or more than two LIVE cell incidences meet at it — the
count counts occurrences, so a cell reaching the piece from both sides states
two, and a coplanar duplicate states none because its survivor already carries
their shared cell. A piece fences WITHOUT a fan when the input edge is
non-manifold (`plane_edge_non_manifold_flag`) or the cells it bounds differ in
depth. Everything else is `crossable`, and a piece no cell names fences nothing.

Two joins cross between the arrangement's spaces and the exposed stream:
`global().exposed_parent_of()[e * 3 + s]` gives the piece ticket of the edge
corner `s` of exposed triangle `e` opens (`-1` when the edge lies on no piece),
and `row_of()` / `exposed_of_row()` map between the stream and the arrangement
rows.

### Openness triage: the fence chooses the branch

An open boolean result has three possible authors — the arrangement, the
triangulation, or the classification. The fence verdict on the result's own
boundary edges separates them before any other investigation. For each boundary
edge, resolve its piece through `exposed_parent_of()` and read:

1. **The piece is `crossable`.** The flood joined across it, so classification
   decided both sides alike. Either the fence is right and the arrangement never
   stated the seam (check the instance flags in step 3), or the fence is wrong —
   the depth reading or the non-manifold fact is missing.
2. **The piece carries a `fan`.** The radial tier owns the edge.
   `descriptor().fans.pieces[f]` names the piece of fan `f`, ascending, so one
   binary search finds the fan; its pages are `fans.page_offsets` over
   `fans.rows`, with `fans.dirs` the side each occurrence sits on. Then check
   `descriptor().valid[f]`: a lone page has no cyclic ring to pair, is counted
   in `n_invalid_fans`, and is deliberately never walked. An unwalked fan leaves
   the two sides unmerged.
3. **The piece is `-1`.** The edge is interior to some triangulation — the
   emission, the weld compaction, or the exposure dropped a row. This is a plane
   tier defect, not a classification one; go back to step 4's laws.

Do not read a boundary edge's absence from the fan tables as evidence of a
T-junction. It only says the edge is interior to some triangulation, which a
dropped row produces just as readily as an unbroadcast vertex does. Distinguish
them by testing exactly, on the lattice, whether a created point actually lies
on the segment.

Coplanar contacts are not a parity anomaly to be explained away: a collapsed
wall is one surface shared by two forms. Before calling anything odd, read
`dead()`, `stacked()` and `coplanar_triples()` on the graph — a dead triangle
carries its survivor's cell, and the stack table is sorted by survivor so a
survivor's partners are contiguous.

## Step 6: classification

`csg_graph::labels()` is two tiers over disjoint carriers compacted into one
dense component id space: a CUT CCL over the cells that crosses a piece only
where the fence allows, and a SURFACE CCL over the faces the arrangement never
cut, flood-filled through the prebuilt `manifold_edge_link`, the two bridged
across the source mesh's manifold edge.

- `open_component_mask()` must be all zero for closed operands. Intersecting
  closed meshes cannot produce an open fragment, so a set bit is an arrangement
  defect on its own — not something to repair in classification.
- `triangle_labels()[e] == none_label` on a live (non-`dead()`) triangle means
  the flood never reached it.
- `polygon_labels(tag)[face] == none_label` says the face was classified CUT, so
  its triangles carry the labels instead. A face that is `none_label` AND whose
  `slot_range(slot)` is empty carries neither — that is the silent-hole
  signature of step 0. Re-read `failed()`.

The descriptor tier answers the volumetric questions:

- `descriptor().domain_of_side[2c + s]` — the domain on side `s` of component
  `c`; `bundle_of_component`, `tag_of_component`, `bundle_to_tags`;
- `descriptor().fans` and `.valid` — one fan per fan piece, its pages radially
  ordered, admitted by the fence's `fan` verdict. The ring turns around the
  piece's own carrier line, which
  `tf::intersect::graph::make_plane_edge_radial_authority` reads off every
  definition of the piece; `fans.n_refused` counts the pieces whose
  definitions named no single line — distinct edges welded into one canonical
  identity — which keep a ring read off their pages' coordinates;
- `domain_volumes()` — exact signed 2x volumes in lattice units, the seeder's
  oracle; the most negative entry is the unbounded universe, and the outer shell
  is the boundary between it and everything else;
- `inclusion()` — per-domain operand bitvectors, blocked per domain by
  `make_range()`;
- `tf::csg::graph::compute_chosen_sides` — per component `1` / `0` / `-1`, the
  last meaning both sides have the same membership so the component contributes
  no boundary face;
- sheets: `is_sheet()`, `sheet_folds()` (`{component, sheet tag, reversed}` per
  coplanar fold) and `anchor_sheet_sides`;
- `domain_nesting_merges()` — repairs contact-free nested shells, consumed by
  `make_csg_domains` only and irrelevant to the boolean mesh read.

## Step 7: raw output through provenance

When the emitted mesh is open or non-manifold:

1. print the exact bad edge ids and all incident output faces;
2. use `tf::return_source_ids` (tag labels + face labels) to name the selected
   producers;
3. use `tf::return_index_map` (`mesh_arrangement_index_map`,
   `csg_domains_index_map`) to translate output point ids back to original or
   created identities;
4. resolve a created id in `created_points()`; `vertex_offsets()` and
   `tag_of_flat(id)` say which form a flat original id belongs to;
5. inspect those identities in the corresponding plane spans. A face's exposed
   slot is `global().face_slot_base()[tag] + object`; `slot_range(slot)` is its
   `[begin, end)` triangle range in the stream, and `row_of()` carries each of
   those triangles back to its arrangement row.

Do not locate producers by searching for equal output coordinates. Run this
check BEFORE `cleaned()` — cleanup can hide the defect by welding points or
removing duplicate faces, and that is not proof the pipeline was correct.

## Coordinates: read the lattice, not the deconversion

The pipeline's coordinates are the exact integer lattice
(`tf::exact::resolve_int_type` of the input width: `int32` for float, `int64`
for double). Every entry that emits geometry takes an output coordinate type,
and an INTEGRAL one keeps the answer on the lattice the arrangement actually
ran on — `make_csg_mesh`, `make_intersection_curves`, `make_arrangement_mesh`,
`make_mesh_arrangements`, `make_polygon_arrangements`, `make_outer_shell`. An
entry that owns its graph hands back the converter alongside the geometry; an
entry that reads a built graph does not, because the graph holds it.

A check whose subject is a geometric fact of the arrangement —
self-intersection, watertightness, seam-edge identity — asks for integer output
and compares exactly. A re-intersection check may seed directly with that
integer mesh, since an integral input coordinate passes through the converter
unchanged. Deconversion to the input real is an output property, not the fact
under test: it moves coordinates and manufactures crossings the arrangement does
not contain, and a float-grid check then needs an arbitrary threshold and
reports defects that are not there.

## The harnesses

`local_tests/` holds the harnesses whose fixtures are not in the repo. Each runs
float and double in one process and returns the combined failure count as its
exit code; each occupies the machine, so run one at a time under an alarm.

- **`regress_domains`** — the domain and boolean product across opposing and
  aligned sheets, sphere-within, geological six-way, bunny two paths, nested
  spheres, the vein system, the offset-bunny ladder, splitter x roi, and
  coplanar-reversed x tolerance. `TF_TRI=refined` runs the csg-graph sections
  through the refined triangulation. Its outer-shell section reads the shell on
  the lattice and checks watertightness with `make_boundary_paths` +
  `make_self_intersection_curves` on the integer mesh; the float residue is
  reported as an informational property, not a failure.
- **`regress_triangulations`** — the exposed triangle stream and its surfaces:
  the DFN origin case, geological, the corpus refusal pairs, coplanar stacks,
  the exposure surface, the opt-in full corpus, the refined quality floors, and
  the DFN curves oracle. All four output routes are read as lattice integers,
  with exact integer curve-edge keys, so no check carries a measurement scale.

`regress_domains` pins come from the public baseline at the same width unless a
comment records otherwise; `regress_triangulations` has no public baseline — its
subject is a surface the public tree does not have — so its pins are this tree's
own and labelled as such. Integer and topology pins are width-independent;
volume and ulp-sensitive pins are specialised per width with their ground-truth
source in the comment.

`probe_seam_vs_fence` is the standing conformance probe between the curve
extraction's own seam verdict and `make_plane_piece_fences` — it joins the two
edge by edge through `global().exposed_parent_of()`.

## Order of investigation

1. Reproduce at tolerance `0`.
2. Verify the operands are closed and manifold (`tf::is_closed`,
   `tf::is_manifold`).
3. Find the first failing operation in an iterated sequence.
4. Read `graph.failed()` / `csg_graph.failed()`. Non-empty ends the search at
   the plane tier — go to *Step 4: the plane arrangement*, not to
   classification.
5. Read the three censuses to learn which stages did work.
6. Print the complete intersection records for the smallest failing face pair.
7. Follow those identities into the definition table and the plane's edge block.
8. Compare a plane's constraint set with its emitted triangles; check the
   published laws and the `null_refined` oracle if refinement is involved.
9. For an open result, read the fence verdict on each boundary edge to choose
   the branch.
10. Check component labels, the descriptor's domains, and the chosen sides.
11. Translate raw output ids back through source ids, index maps, and
    `created_points()`.
12. State the first broken transition before proposing a fix.

The proof names exact records, identities, plane / piece / cell ids, and
descriptors, and it is stated on the lattice. Geometry evaluates predicates
inside the algorithms; it does not reconstruct topology during debugging.
