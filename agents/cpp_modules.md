# C++ Module-by-Module API Reference

> **Lookup only.** Agents can and should grep current declarations and callers.
> Do not infer implementation strategy from this inventory. Read
> `cpp_performance_philosophy.md` and `cpp_execution_patterns.md` for the
> cross-module execution style, and verify every API here against current code.

Common surface conventions include `snake_case` naming, the `build()` pattern
for stateful structures, `make_*()` factories, and `tf::none_t` deduction.

---

## 1. topology/

### Core Data Structures

| Structure | Template | Build Input | Key Accessors |
|-----------|----------|-------------|---------------|
| `half_edges<Index>` | Index type | `polygons` or `faces + face_membership` | `next()`, `previous()`, `opposite()`, `rotated()`, `half_edge_handles()`, `edge_handles()` |
| `face_membership<Index>` | Index type | `polygons` or `faces + n_unique_ids` | `[vertex_id]` → range of face IDs |
| `manifold_edge_link<Index, N>` | Index, face size | `faces + face_membership` | `[face_id]` → range of `manifold_edge_peer` |
| `vertex_link<Index>` | Index type | `faces + face_membership` or `edges` | `[vertex_id]` → range of neighbor vertex IDs |
| `face_link<Index>` | Index type | `face blocks + face_membership` | `[face_id]` → range of adjacent face IDs |
| `edge_membership<Index>` | Index type | `edges + n_unique_ids + orientation` | `[vertex_id]` → range of edge IDs |

All have `build()` method + `make_*()` free function equivalents.

### Analysis Functions

| Function | Input | Return | Description |
|----------|-------|--------|-------------|
| `make_boundary_edges(polygons)` | polygons | `blocked_buffer<Index, 2>` | Edges shared by exactly 1 face |
| `make_non_manifold_edges(polygons)` | polygons | `blocked_buffer<Index, 2>` | Edges shared by 3+ faces |
| `is_manifold(polygons)` | polygons | `bool` | True if every edge shared by ≤2 faces |
| `is_closed(polygons)` | polygons | `bool` | True if no boundary edges (watertight) |
| `euler_characteristic(polygons)` | polygons | `int` | V - E + F, each undirected edge counted once through the manifold edge link's representative, so boundary and non-manifold edges count like interior ones |
| `connect_edges_to_paths(edges)` | edges | `offset_block_buffer<Index, Index>` | Assemble edges into continuous paths |
| `make_boundary_paths(polygons)` | polygons | `offset_block_buffer<Index, Index>` | Boundary edges assembled into paths |
| `find_eulerian_paths(edges, link, ...)` | edges + link | offsets + edge IDs (output params) | Hierholzer's edge-disjoint path decomposition |
| `label_connected_components(labels, applier)` | label buffer + neighbor callback | `int` (n_components) | Parallel union-find component labeling |
| `make_manifold_edge_connected_component_labels(polygons)` | polygons | `connected_component_labels<Index>` | Component labeling via manifold edge adjacency |
| `orient_faces_consistently(polygons)` | polygons | `bool` (modifies in place) | Consistent face orientation via weighted voting (by face count on an integral coordinate type, whose lattice cannot hold a squared area). Every reversal is decided against the input winding and applied after the walk, so one call settles every orientable manifold-edge component; a component whose parity contradicts is left untouched and the call returns `false` |

### Planar Graph Processing

| Structure | Build Input | Purpose |
|-----------|-------------|---------|
| `planar_graph_regions<Index, Int>` | directed edges + 2D points | Extract minimal closed regions by polar angle walk |
| `face_hole_relations<Index, Int>` | faces + holes + points | Parent-child nesting between faces and holes |
| `face_split_by_edges<Index, Int>` | face loop + edges + points | Subdivide a face by interior edges |
| `hole_patcher<Index, Int>` | outer loop + holes + points | Bridge holes into outer boundary |
| `planar_embedding<Index, Int>` | directed edges + points | High-level: regions + hole relations |

`Int` has no default on any of these: `build` accepts any coordinate type, so
the class has nothing to resolve the lattice from and the caller states it.

### Triangulation

| Symbol | Purpose |
|--------|---------|
| `cdt_region_mode` | What a constrained build's `region_labels()` state: `nesting` (parity of the walls crossed from outside) or `components` (id of the wall-cut component, `0` the hull exterior). Trailing argument of `constrained_delaunay_triangulator::build` / `build_regions` / `build_from_constraints` and `cdt_refiner::build`; `make_cdt` always builds `nesting` |
| `unconstrained_delaunay_triangulator<Index, Coord, Int, VertexPolicy, ExecutionPolicy>` | Point-only divide-and-conquer Delaunay retaining no constraints, region labels, or editable adjacency; backs `make_cdt(points)`. `build(points)` returns `bool`; `faces()`, `n_unique_points()`, `unique_input_id()`, `converted_unique_point()`, `take_index_map()` |

`constrained_delaunay_triangulator<Index, Coord, Int, ExecutionPolicy>`,
`unconstrained_delaunay_triangulator` and `cdt_refiner<Index, Coord, Int>`
default `Int` from `Coord` via `tf::exact::resolve_int_type` — int32 for float,
int64 for double, identity for an integral `Coord`.

### Half-Edge Sentinel Values
- `-1`: boundary
- `-2`: non-manifold
- `-3`: orientation fault
- `-4`: removed

---

## 2. spatial/

### Tree Types

| Type | Alias | Bounding Volume |
|------|-------|----------------|
| `tree<Index, aabb<RealT, Dims>>` | `aabb_tree<Index, RealT, Dims>` | Axis-aligned bounding box |
| `tree<Index, obb<RealT, Dims>>` | — | Oriented bounding box |
| `tree<Index, obbrss<RealT, Dims>>` | — | OBB + RSS hybrid |
| `mod_tree<Index, BV>` | — | Modifiable tree (main + delta) |

Build: `tree.build(primitives, config_tree(inner_size, leaf_size))`

### Queries (all require tagged forms: `form | tf::tag(tree)`)

| Function | Input | Return | Description |
|----------|-------|--------|-------------|
| `neighbor_search(form, query)` | form + point/segment/ray | `tree_metric_info` | Nearest primitive |
| `neighbor_search(form, query, radius)` | + linear radius | `tree_metric_info` | Within radius |
| `neighbor_search(form, query, knn)` | + k-NN container | fills knn | k-nearest neighbors |
| `neighbor_search(form0, form1)` | two forms | `tree_metric_info_pair` | Closest pair between two meshes |
| `distance(form, query)` | form + query | `RealT` | Euclidean distance |
| `distance2(form, query)` | form + query | `RealT` | Squared distance |
| `ray_cast(ray, form, config)` | ray + form | `tree_ray_info` | First ray-primitive intersection |
| `ray_hit(ray, form, config)` | ray + form | `tree_ray_info` | Ray intersection with hit point |
| `intersects(form0, form1)` | two forms | `bool` | Any intersection exists |
| `intersects(form, query)` | form + primitive | `bool` | Primitive intersects form |
| `gather_ids(form0, form1, pred, out)` | two forms + predicate | output iterator | Collect intersecting ID pairs |
| `gather_self_ids(form, pred, out)` | form + predicate | output iterator | Self-intersecting ID pairs |
| `search(form, check_bv, apply)` | form + callbacks | `bool` | Generic tree traversal |
| `search_self(form, check_bvs, apply)` | form + callbacks | `bool` | Self-intersection traversal |

### Result Types
- `tree_metric_info<Index, Info>` — element ID + metric info, `operator bool()` for validity
- `tree_metric_info_pair<I0, I1, Info>` — pair of IDs + metric point pair
- `tree_ray_info<Index, Info>` — element ID + ray intersection info
- `nearest_neighbors<RandomIt>` — k-NN container, `make_nearest_neighbors(begin, k)`

### Dual-Tree Parallelism
Dual-tree searches use TBB `task_group` with `parallelism_depth` (default 6). Callbacks must be thread-safe — use `tf::local_vector` for aggregation.

---

## 3. geometry/

### Mesh Generation

| Function | Return | Description |
|----------|--------|-------------|
| `make_box_mesh<Index>(w, h, d)` | `polygons_buffer<Index, RealT, 3, 3>` | Axis-aligned box (12 triangles) |
| `make_sphere_mesh<Index>(r, stacks, segs)` | `polygons_buffer<Index, RealT, 3, 3>` | UV sphere |
| `make_cylinder_mesh<Index>(r, h, segs)` | `polygons_buffer<Index, RealT, 3, 3>` | Cylinder with caps |
| `make_plane_mesh<Index>(w, h, w_ticks, h_ticks)` | `polygons_buffer<Index, RealT, 3, 3>` | Subdivided XY plane |
| `make_tube_mesh<Index>(curve, radius, segs)` | `polygons_buffer<Index, RealT, 3, 3>` | Tube around curve |

### Normal & Curvature

| Function | Return | Description |
|----------|--------|-------------|
| `compute_normals(polygons)` | `unit_vectors_buffer` | Per-face normals |
| `compute_point_normals(polygons)` | `unit_vectors_buffer` | Per-vertex area-weighted normals |
| `compute_principal_curvatures(...)` | `curvature_values` or `curvature_full` | k1, k2 via quadric fitting |
| `compute_shape_index(polygons)` | buffer | Shape classification from curvatures |

### Registration

| Function | Return | Description |
|----------|--------|-------------|
| `fit_rigid_alignment(X, Y)` | `transformation` | SVD-based rigid (Kabsch) or point-to-plane |
| `fit_similarity_alignment(X, Y)` | `transformation` | Procrustes with uniform scale; a tagged frame moves both the covariance and the source spread into world space |
| `fit_knn_alignment(X, Y, state, config)` | `transformation` | Gaussian-weighted k-NN soft correspondence |
| `fit_obb_alignment(X, Y, sample_size)` | `transformation` | OBB alignment with ambiguity resolution |
| `fit_icp_alignment(X, Y, state, config)` | `transformation` | Iterative closest point with convergence |

### Processing

| Function | Return | Description |
|----------|--------|-------------|
| `triangulated<Index>(polygons)` | `polygons_buffer<Index, ..., 3>` | The mesh triangulation tier (`arrangement/mesh/`) read as a mesh — the ONE public triangulation entry: each face triangulated on its own boundary, a shared edge one identity in both, corners in the source winding; a face needing resolution is resolved and MINTS, so the points are the input's then this call's mints. Takes an indexed mesh, a single `tf::polygon`, or a SOUP — a soup is `tf::cleaned` to shared-vertex identity first, so no machinery below the entry ever sees one. `Index` answers two questions: the WIDTH the faces are written in (default: the input's own — ask for `int32_t` when an `int64_t` mesh does not need it) and, for a soup, the NAME of an output whose input carries no index type (default `int`). `tf::return_refused` adds the faces that refused every round, in the INPUT's index type; it takes indexed meshes only, a soup having no face identity that survives the clean |
| `laplacian_smoothed(points, iters, lambda)` | `points_buffer` | Laplacian smoothing |
| `taubin_smoothed(points, iters, lambda, kpb)` | `points_buffer` | Volume-preserving smoothing |
| `ensure_positive_orientation(polygons)` | `bool` (in-place) | Outward-facing normals; `false` and no volume flip when any component is not orientable |
| `make_sharp_edges(polygons, angle)` | `blocked_buffer<Index, 2>` | Edges exceeding dihedral threshold |
| `make_curve_frames(curve, T, N, B)` | void (output params) | Parallel transport frames |
| `chamfer_error(A, B, outlier_pct)` | `RealT` | Mean nearest-neighbor distance |

---

## 4. remesh/

### High-Level Operations

| Function | Modifies | Description |
|----------|----------|-------------|
| `isotropic_remesh(he, points, frame, config)` | half_edges + points | Split/collapse/flip/relax loop |
| `decimate(he, points, target_proportion, config)` | half_edges + points | QEM-based edge collapse to target face count |

### Primitive Operations

| Function | Description |
|----------|-------------|
| `split_long_edges(he, points, max_length)` | Split edges exceeding length threshold |
| `split_edges(he, points, handler)` | Generic edge splitting via handler |
| `collapse_short_edges(he, points, min_length)` | Collapse edges below length threshold |
| `collapse_edges(he, points, handler)` | Generic edge collapse via handler |
| `optimize_valence(he, deviation, iters)` | Flip edges to optimize vertex valence (target: 6 interior, 4 boundary) |
| `tangential_relaxation(he, points, workspace, iters, lambda)` | Move vertices toward centroid, project onto tangent plane |
| `equalize_areas(he, points, workspace, iters, lambda)` | Minimize squared area variation in 1-ring |
| `make_feature_mask(he, points, angle)` | Mark sharp edges and classify vertices (regular/crease/corner) |

All remesh operations work on mutable `half_edges<Index>` + `points_buffer` directly. Feature-aware variants skip crease edges and corner vertices.

---

## 5. intersect/

### Pipeline Structures

| Structure | Build Input | Output |
|-----------|-------------|--------|
| `polygon_intersections<Index, RealT, Int>` | 1, 2, or N tagged polygons + config | Intersection records whose ids are canonical point names — no coordinate table; self records when the mode carries `self_intersections` (implied for the one-form build) |
| `intersections_within_segments<Index, RealT, Dims, Int>` | tagged segments (tree + edge membership) | Segment-segment intersection records + an int-coordinate point table |

Both default `Int` from `RealT` via `tf::exact::resolve_int_type` — int32 for
float, int64 for double, identity for an integral `RealT`.

`tf::polygon_intersections` is the intersections-between-polygons of the
library; arity plus `intersect_config` decide between/within, and there are no
alias types. It is not exported by `intersect.hpp` — include
`intersect/polygon_intersections.hpp` directly. Point identity is its surface:
`n_vertex_points()` splits the id space into vertex anchors (`vertex_anchor`)
and edge-carried classes (`home_edge` + `exact_parameter`), with
`edge_carriers()` / `edge_splits()` per carrier.

### High-Level Functions

| Function | Return | Description |
|----------|--------|-------------|
| `make_intersection_curves(poly0, poly1, mode)` | `curves_buffer<Index, RealT, 3>` | Connected intersection curves between two meshes |
| `make_intersection_curves(range_of_forms, mode)` | `curves_buffer` | All pairwise intersection curves for N meshes |
| `make_self_intersection_curves(polygons, mode)` | `curves_buffer` | Self-intersection curves |
| `make_intersection_edges(si, faces)` | `blocked_buffer<Index, 2>` | The chords of every `(face, cut)` group as edge pairs; `tf::intersect::for_each_cut_group` finds the groups and `for_each_cut_chord` states one group's chords (`intersect/records/`), both shared with the iso module |

### Intersection Modes (`intersect_mode`, bitwise)
- `sos` — SoS fan triangulation (all edge-face records)
- `primitives` — Full 5-type classification (EF/EE/VE/VF/VV)
- `resolve_crossing_contours` — Inter-contour crossings. Declarative: such a
  pair needs a third tag to exist, so the graph derives it from arity and never
  reads the flag
- `resolve_self_crossing_contours` — Resolve intra-contour crossings
- `resolve_contours` — both resolve bits
- `self_intersections` — atomic bit: also generate each form's self records
- `within` — `self_intersections | resolve_self_crossing_contours`; what callers
  write, since a self contour class only has self-crossings

---

## 6. arrangement/

### The Arrangement

| Function / Structure | Return | Description |
|----------|--------|-------------|
| `make_arrangement_graph<Int>(operands, config)` | `arrangement_graph<Policy, Int>` | The build. Operands: a range of forms, one form (its self arrangement, `within` implied), or two forms of possibly different types. Completes any missing tree / face membership / manifold edge link in parallel |
| `arrangement_graph<Policy, Int>` | — | The arrangement of a set of forms, everything below classification: exposed triangle stream, cells, piece incidence, piece fences, coplanar stacks, `created_points()` |
| `arrangement_config` | — | `{intersect_config intersect, triangulation_type triangulation}`; implicitly constructible from either alone. Default intersect = `primitives \| resolve_crossing_contours` |

### Arrangement Products

| Function | Return | Description |
|----------|--------|-------------|
| `make_mesh_arrangements(poly0, poly1)` | `(mesh, tag_labels, face_labels)` | Decompose into classified regions |
| `make_mesh_arrangements(range_of_forms)` | `(mesh, tag_labels, face_labels)` | N-mesh arrangements |
| `make_polygon_arrangements(polygons)` | `(mesh, face_labels)` | Split one mesh at its self-intersection curves |
| `make_arrangement_mesh(graph)` | `polygons_buffer` | Materialise the full arrangement mesh; `return_source_ids` / `return_index_map` overloads |
| `make_intersection_curves(graph)` | `curves_buffer` | Seam polylines read off the arrangement |
| `make_segment_arrangements(segments)` | `(segments, edge_labels)` | Split segments at all intersection points (own tier: `tf::intersect::segment_intersection_graph`) |
| `make_stitch_index_map(arrangement_map)` | `stitch_index_map` | Read an arrangement result in stitching terms, so operand topology and trees carry onto it |

`make_mesh_arrangements` and `make_polygon_arrangements` return
`mesh_arrangement_index_map` / `polygon_arrangement_index_map` under
`tf::return_index_map`, and curves under `tf::return_curves`.

### Template slots (arrangement + csg, one law)

An entry that CONSTRUCTS the arrangement — the builders and the fused form
entries — takes the lattice `Int` FIRST, then `OutputCoordinateType` second
where it also emits geometry. An entry that READS a built graph takes only
`OutputCoordinateType`: the graph's type already fixed the `Int` and the graph
holds the converter. So `make_arrangement_graph<Int>` / `make_csg_graph<Int>`,
`make_mesh_arrangements<Int, OCT>` / `make_polygon_arrangements<Int, OCT>` /
`make_outer_shell<Int, OCT>(polygons)`, versus `make_csg_mesh<OCT>(graph)` /
`make_arrangement_mesh<OCT>(graph)` / `make_intersection_curves<OCT>(graph)` /
`make_outer_shell<OCT>(graph)`. `OCT` is the REQUEST (may be `tf::none_t`);
`RealOut` names the RESOLVED type inside an implementation, and
`tf::resolved_output_real_t<OCT, InputReal>` (`core/resolved_output_real.hpp`)
is the one place that resolution and its two laws are stated. A form entry with
floating-point input and integral output returns the converter alongside the
geometry, so the shape change is loud at the call site.

### Plane Pipeline (`arrangement/planes/`, `tf::arrangement`)

| Structure | Purpose |
|-----------|---------|
| `plane_world<Policy>` | The one carrier the plane arrangement's seam speaks; `make_plane_world(local_arrangement)` borrows the production world |
| `plane_arrangement<Index, Int>` | One triangulation per plane carrier: triangles, slot parents, corner subs, coplanar stacks, optional per-triangle cells |
| `make_plane_arrangement_cells(arrangement)` | `plane_arrangement_cells<Index>` — the 2-cells numbered densely across the planes |
| `make_plane_piece_incidence(arrangement, cells)` | `plane_piece_incidence<Index>` — piece <-> cell incidence from both sides |
| `make_plane_piece_fences(arrangement, world, incidence, cells)` | `plane_piece_fences` — per piece: `fan` and `crossable`, the one producer of both verdicts |

---

## 7. iso/

The scalar-field pipeline: a separate tier from the arrangement, driven by
field crossings rather than by polygon intersections. Umbrella
`trueform/iso.hpp`; the machinery is `tf::iso` under `iso/cut/`.

### Public Surface

| Function / Structure | Return | Description |
|----------|--------|-------------|
| `scalar_field_intersections<Index, RealT, Dims>` | — | polygons + per-vertex scalars + threshold(s) -> field-crossing records + real-coordinate points (`build`, `build_many`); the sole producer of field-point identity |
| `make_isocontours(polygons, scalars, cut_value[s])` | `curves_buffer` | Scalar-field isocontours |
| `embedded_isocurves(polygons, scalars, cut_values)` | `(mesh, labels, face_labels)` | The whole mesh recut, every band kept; `tf::return_curves` / `tf::return_refused` overloads |
| `make_isobands(polygons, scalars, cut_values, selected)` | `(mesh, labels, face_labels)` | The selected bands alone; same overload set |

### The Cut Tier (`iso/cut/`, `tf::iso`)

| Function / Structure | Purpose |
|-----------|---------|
| `build_iso_cuts<Index>(polygons, scalars, cut_values)` | The dispatcher: `(scalar_field_intersections, iso_cut_regions, partition_ids)` |
| `cut_iso_faces<Int>(polygons, si, category, on_cut, created_cut, bands)` | Cuts every crossed face, splitting by state: the one-chord face walks its chain, everything else takes a constrained Delaunay build in `cdt_region_mode::components` whose components are the pieces |
| `split_iso_face_chord<LabelType>(local, object, n_original, face_size, ...)` | The one-chord state: the chord's two chain positions separate the chain into the two pieces, each fanned. Declines (returns false) for any arity but a triangle, a degenerate projection, a chord bounding no area, and any chord count but one |
| `prepare_iso_face_cut<Index, Int>(records, face, n_original, field_points, axes, local, point_of_flat)` | The face's boundary chain in flat identities — which IS the point table, position being the local index — its chain edges plus interior chords as the constraint set, and the chain's own winding |
| `emit_iso_face_regions<LabelType>(local, object, polygon, axes, converter, n_original, ...)` | Minimal-flat-id readback, winding against the prepared chain, band per region, triangles grouped by region (region 0 — hull exterior and pockets — dropped) |
| `iso_band_of_triangles<LabelType>(triangles, n_original, ...)` | The band a piece lies in — the one producer of that fact for both states |
| `gather_iso_band_triangles<Index>(band_regions, regions, triangles, origins, band_offsets)` | The kept pieces' triangles band-major, by counts + one prefix + disjoint writes; `band_offsets` in triangle units |
| `iso_cut_regions<Index, RealT, Dims>` | `triangles` blocked per region, `faces`, `minted_points`; corners are flat ids in `[originals \| field points \| minted]` |
| `make_surface_scalar_labels<LabelType>(polygons, si, category)` | Band per uncut face; `-1` marks a cut face |
| `embedded_isocurves<LabelType>` / `make_isobands<LabelType>` | The two emitters, both reading the one region product |

---

## 8. exact/

### Integer Types
`int32` (std), `int64` (std), `int128` (compiler-specific), `int256` (custom)

`meta<Int>`: `T0` = self, `T1` = double width, `T2` = quadruple width

### Predicates

| Function | Return | Description |
|----------|--------|-------------|
| `orient3d_sos(vertices[4])` | `bool` | 3D orientation with SoS (never zero) |
| `orient3d_sign(p0, p1, p2, p3)` / `orient3d_sign(vertices[4])` | `int` (-1/0/+1) | 3D orientation exact sign |
| `orient3d_value(p0, p1, p2, p3)` | `T2` | Signed volume × 6, exact |
| `orient2d(p0, p1, p2)` | `T2` | Signed area × 2, exact |
| `orient2d_sos(v0, v1, v2, ax0, ax1)` | `bool` | 2D orientation with SoS |
| `signed_area_2x(polygon)` | `T2` | Polygon signed area × 2 |
| `signed_area_sign(polygon)` | `int` | +1 (CCW), -1 (CW), 0 |
| `projection_axes(p0, p1, p2)` / `projection_axes<Int>(normal)` | `pair<int,int>` | Best 2D projection axes, from three points or from an exact normal |
| `plane_support<Int>` | struct | The points a carrier stands on: `offer(p)` takes points in the caller's own order and keeps the first, the first distinct, and the first off their line, with the accepted cross as `normal`. The one producer of that scan; `make_supported_plane_frame` (offer-shaped), `tf::intersect::graph::plane_face_support` (face-shaped) and `make_face_plane` read it |
| `classify_segments(a0, a1, b0, b1, ...)` | `optional<pair<contacts>>` | Full segment-segment contact classification |

### Coordinate Conversion
- `pt_converter<Int, RealT, Dims>` — real → the `Int` lattice (99% of its max range)
- `vertex_converter<Int, RealT, Dims>` — pt_converter + per-mesh globally unique vertex IDs
- `resolve_int_type<Int, Coord>` — the standing law: `tf::none_t` becomes int32
  for float, int64 for double, `Coord` itself for an integral `Coord`. Every
  factory (`make_pt_converter`, `make_vertex_converter`,
  `tf::make_exact_coordinate_converter`, `tf::make_exact_points`) leaves both
  `Int` and `RealT` unstated by default and reads them off its input
- `input_lattice<Index, RealT, Int>` — where every original vertex of every
  operand stands on the lattice, built once over the union of the operands:
  the shared converter, the flat vertex space that names `(tag, id)` by one
  integer, and — when a tolerance is given — the table the door
  (`exact/door/`) placed every vertex into. A tolerance of zero is the
  identity: no table, no face read, the plain converter
- `input_lattice_reader<Index, RealT, Int, ApplyToForm>` — the one reader of an
  original vertex's position, bound to the caller's own forms: `(tag, id)` or a
  flat id answers from the placed table when there is one and from the
  converter over the input's own coordinate when there is not

---

## 9. clean/

| Function | Input | Return | Description |
|----------|-------|--------|-------------|
| `cleaned(polygons)` | polygons | `polygons_buffer` | Remove degenerate faces + duplicate vertices |
| `cleaned(polygons, tolerance)` | + tolerance | `polygons_buffer` | Tolerance-based deduplication |
| `cleaned(segments)` | segments | `segments_buffer` | Remove zero-length edges |
| `cleaned(points)` | points | `points_buffer` | Deduplicate points |
| `cleaned(curves)` | curves | `curves_buffer` | Deduplicate + reconnect paths |

Index-map return shapes depend on the carrier: polygons and segments return
primitive and point maps, points return a point map, curves return only a point
map because paths reconnect, and soup cleaning has no source identity map.

Low-level: `make_clean_index_map(geometry)` returns maps without applying them.

---

## 10. reindex/

| Function | Description |
|----------|-------------|
| `reindexed_by_ids(geometry, ids)` | Keep elements at specified indices |
| `reindexed_by_mask(geometry, mask)` | Keep elements where mask is true |
| `reindexed_by_ids_on_points(geometry, point_ids)` | Keep faces whose ALL vertices are in point ID list |
| `reindexed_by_mask_on_points(geometry, point_mask)` | Keep faces whose ALL vertices pass mask |
| `reindexed(geometry, face_map, point_map)` | Apply pre-computed index maps |
| `split_into_components(geometry, labels)` | Split by label → `(vector<buffer>, vector<label>)` |
| `concatenated(geom0, geom1, ...)` | Merge multiple geometry collections |
| `make_dynamic(polygons_buffer)` | Convert fixed-size to dynamic-size polygons |

All `reindexed_*` functions have `return_index_map` variants.

---

## 11. io/

| Function | Return | Description |
|----------|--------|-------------|
| `read_stl(path)` | `polygons_buffer<int, float, 3, 3>` | Binary/ASCII STL with auto-deduplication |
| `write_stl(polygons, path)` | `bool` | Binary STL (<500MB buffered, >500MB streamed) |
| `read_obj(path)` | `polygons_buffer<int, float, 3, dynamic_size>` | ASCII OBJ (1-based → 0-based indices) |
| `read_obj<3>(path)` | `polygons_buffer<int, float, 3, 3>` | OBJ with fixed triangle faces |
| `write_obj(polygons, path)` | `bool` | ASCII OBJ (parallel two-pass write) |

Memory buffer variants accept `range<const char*, dynamic_size>` for in-memory parsing.

## 12. csg/

Build one arrangement of N operands, answer arbitrarily many boolean
expressions against it. This supersedes chained `make_boolean` calls for
anything N-ary or multi-query.

| Function | Return | Description |
|----------|--------|-------------|
| `make_csg_graph(operands[, sheets][, arrangement_config])` | `csg_graph<Policy, Int, Arrangement>` | The build: `make_arrangement_graph` (same operand shapes) + the classification tier (components, radial fans, descriptor, domain inclusions, domain volumes, sheet anchoring) |
| `tf::csg::op(i)` | `csg::expr` | Expression leaf; combine with `\|`, `&`, `-`, `~` |
| `tf::csg::selection(tags[, expr])` / `tf::csg::inside(tags, expr)` | `csg::selection_t` | What `make_csg_mesh` emits of the named forms: the expression's boundary (`selection`, exactly one side in the region, wound outward), the surface inside its region (`inside`, both sides in it, stored winding), or the embedded read (no expression). `selection_kind { boundary, inside }`; `make_csg_domains` takes a `boundary` selection only |
| `make_csg_mesh(graph, expr)` | `polygons_buffer` | Boolean result mesh; overloads: no-expr (full arrangement mesh), `tf::return_source_ids` (+ tag/face labels), `tf::return_index_map` (+ `mesh_arrangement_index_map`) |
| `make_csg_domains(graph[, expr][, domain_config])` | `(cells, ids)` | One watertight mesh per kept domain; same `return_source_ids` / `return_index_map` overload set (`csg_domains_index_map`) |
| `make_outer_shell(graph)` | `polygons_buffer` | Structural read: the boundary between the unbounded universe and everything the forms enclose (the most-negative-volume domain is the universe). Takes NO domain flags — both are settled facts before the read. The `polygons` overload repairs a mesh to its outer shell by building its own one-form `csg_graph` and answering the same read, so the shell never round-trips through floats |
| `make_intersection_curves(graph)` | `curves_buffer` | Cross-tag seam polylines read off the graph (coincident walls excluded) |

### Face Classification (`arrangement_class`, `csg/arrangement_class.hpp`)
- `none`, `inside`, `outside`, `aligned_boundary`, `opposing_boundary`,
  `on_boundary`

### Boolean Operations

The two-operand entry points: one graph, one expression, answered in one
call.

| Function | Return | Description |
|----------|--------|-------------|
| `make_boolean(poly0, poly1, op)` | `(mesh, labels, face_labels)` | Union/intersection/difference |
| `make_boolean(..., return_curves)` | `+ curves_buffer` | Also returns intersection curves |
| `make_boolean(..., return_index_map)` | `+ mesh_arrangement_index_map` | Output-to-input maps for points and faces |
| `make_boolean(..., sheets)` | as above | Operands in `sheets` bound no volume; they cut as oriented separators |

`boolean_op`: `merge` (union), `intersection`, `left_difference` (A-B),
`right_difference` (B-A).

`tf::triangulation_type{cdt, refined_cdt}` — carried in
`arrangement_config` — selects the cut surface the plane arrangement
emits: plain constrained Delaunay per plane carrier, or Ruppert quality
refinement with boundary splits negotiated through shared dyadic records
(watertight by ids across planes). Sheets (`is_sheet` operands) cut
volumes through the same algebra without enclosing one.

Topology additions backing this: `constrained_delaunay_triangulator`
(incremental Bowyer–Watson/BRIO core, exact int predicates, region-parity
labels, preserved or arranged constraint modes), `cdt_refiner`
(Ruppert refinement with recorded dyadic constraint splits),
`triangulation_type`, `make_cdt(..., split_constraints)`.
