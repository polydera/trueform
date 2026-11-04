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
| `make_boundary_edges(polygons)` | polygons | `blocked_buffer<Index, 2>` | Edges shared by exactly 1 face, face by face and corner by corner |
| `make_non_manifold_edges(polygons)` | polygons | `blocked_buffer<Index, 2>` | Edges shared by 3+ faces |
| `make_non_manifold_vertices(polygons)` | polygons | `buffer<Index>` | The vertices whose faces are not one fan, ascending: an edge at the vertex carries 3+ faces, or its faces fall into pieces meeting at the vertex alone. Walked on the face membership, so no half-edge structure is built; winding never enters and a vertex no face names is not reported |
| `split_non_manifold_vertices(polygons)` | polygons | `polygons_buffer` | One vertex per fan: the fan holding the vertex's smallest face keeps its id, every other takes a minted copy of the coordinates and its corners are rewired onto it. Faces keep their ids, arity and winding; the points are the input's then the mints in id order. An edge 3+ faces carry is crossed by no fan and separating the fans it holds apart would tear it into boundary copies, so a vertex any of whose edges carries 3+ faces is left untouched and `make_non_manifold_vertices` still names it. `tf::return_index_map` adds the point map: per output point the input point it copies, an original itself |
| `is_manifold(polygons)` | polygons | `bool` | True when every vertex's faces are one fan, which also says every edge carries at most two of them. Winding is `orient_faces_consistently`'s fact and never enters this one |
| `is_closed(polygons)` | polygons | `bool` | True if no boundary edges (watertight) |
| `has_holes(polygons)` | polygons | `bool` | True if a boundary rim is a hole: a closed rim, or an open one with an end that carries no edge 3+ faces share. A sheet's free rim is a hole, the answer standing against the expectation of a closed solid |
| `has_flaps(polygons)` | polygons | `bool` | True if a boundary rim is a flap: an open rim each of whose ends carries an edge 3+ faces share, the attachment its faces hang from. Every rim is a hole or a flap, so a mesh with no boundary edges has neither |
| `euler_characteristic(polygons)` | polygons | `int` | V - E + F, each undirected edge counted once through the manifold edge link's representative, so boundary and non-manifold edges count like interior ones |
| `connect_edges_to_paths(edges)` | edges | `offset_block_buffer<Index, Index>` | Assemble edges into continuous paths |
| `make_boundary_paths(polygons)` | polygons | `offset_block_buffer<Index, Index>` | Boundary edges assembled into paths by `find_eulerian_paths`, so a vertex the boundary reaches twice is traversed rather than split and a pinched boundary comes back as one figure eight. The vertex-path read; for the rim as a carrier ask `make_boundary_rims` |
| `make_boundary_rims(polygons)` | polygons | `boundary_rims<Index>` | The boundary as rims: `vertices` and `faces` one offset block each per rim, plus the per-rim `closed` fact. Rim edge `k` runs from vertex `k` to vertex `k + 1` and is carried by face `k` alone, a closed rim of `n` vertices having `n` edges and an open one `n - 1`. A rim ends where the boundary stops passing straight through, so a pinch splits it. The mesh decides the product: the boundary edges are its own order, a rim with two ends precedes the closed ones, and a consistently wound mesh gives rims that run the way its faces wind their boundary |
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
| `delaunay_tetrahedralizer<Index, Coord, Int, ExecutionPolicy>` | Exact incremental 3D Delaunay of integer lattice sites, parallel by default (`serial_delaunay_execution_policy` selects serial). `build(points)` names each site by its input slot, `build(range of site_type)` keeps the caller's names; coincident sites weld to the lowest name and cospherical ties break symbolically by name. `append_sites(sites)` inserts distinct sites carrying fresh names, the first being `next_name()`, after a successful build. `tets()` are the finite cells in one canonical order — ascending by sorted corner slots, each from its smallest corner, positively oriented — the same bytes at any worker count; `neighbors()` (`k_none` across the hull), `sites()`, `index_map()`, `refusal()` (`tetrahedralization_refusal`), `stats()` (`tetrahedralization_stats`), `is_valid()` |

`constrained_delaunay_triangulator<Index, Coord, Int, ExecutionPolicy>`,
`unconstrained_delaunay_triangulator` and `cdt_refiner<Index, Coord, Int>`
default `Int` from `Coord` via `tf::exact::resolve_int_type` — int32 for float,
int64 for double, identity for an integral `Coord`. `delaunay_tetrahedralizer`
takes the same order and default, its `Coord` being the integer a site is
stored in: the lattice itself, or a wider integer holding a scaled lattice,
in which case the caller states `Int`. A `Coord` narrower than `Int` does not
compile.

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
| `make_winding_moments(tree, polygons)` | built tree + its polygons | `winding_moments<RealT>` | Per-node winding expansion to Taylor order two — dipole, mixed second moment, and the quadrupole third-moment tier — folded into four-lane child blocks; tag beside the tree |
| `winding_number(form, point[, config])` / `(form, points, out[, config])` | form with tree + winding moments | `double` / fills `out` | Generalized winding number, far nodes by the folded order-two expansion, near leaves exact; `winding_config{beta = 2}`. A winding-tagged form also switches `signed_distance`'s sign to `w > 0.5` and then needs only the tree |

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
| `compute_face_quality(polygons)` | `face_quality<T>` | Per-face quality as flat buffers: `quality`, `min_angle`, `max_angle`, `aspect_ratio`. The corner angles (unsigned, in `[0, pi]`, so a reflex corner reads its explement) and the aspect ratio (longest side over shortest) hold at any arity; `quality` is the triangle measure `tf::triangle_quality` states, and a face that is not a triangle reads `-1` |
| `compute_dihedral_angles(polygons)` | `dihedral_angles<Index, T>` | Every edge two faces share, once through the manifold edge link's representative: `edges` blocked in vertex pairs beside an aligned `angles` buffer, face-major. The angle is between the face normals, so a flat surface reads `0`; a boundary or non-manifold edge joins no pair and is not stated. Missing link or normals are built for the call |
| `make_geodesic_distances<OCT>(polygons, seeds[, radius])` | `buffer<RealOut>` | Geodesic distance from the nearest seed: a label-correcting front over the two-neighbour unfolding update (`geometry/geodesic/`), run in double and stated in the resolved output type (the mesh's own by default); zero at a seed, infinity unreached or past `radius`; split into one front per thread past 200 points, its determinism contract stated at the entry against `geometry::geodesic_revision_tolerance`; a finite radius forms only the triangle images its front reaches; stored coordinates, membership and vertex link read where tagged |
| `make_dirichlet_field<OCT>(polygons, corner_weights, fixed_ids, fixed_values, order)` | `dirichlet_field_result<RealOut> {buffer<RealOut> field, dirichlet_field_status status}` | One scalar held at `fixed_ids` and vanishing under `dirichlet_order::harmonic` (`L x = 0`) or `biharmonic` (`L M^-1 L x = 0`) elsewhere, over the Laplacian the per-corner weights state (`tf::none` = cotangent); sparse Cholesky in double, refined once against the true residual, the field stated in the resolved output type; `status` names the refusal (`refused_operator`, `refused_unanchored`, `refused_index`, `refused_indefinite`, `refused_residual`) |
| `geometry::laplacian_system` and its passes (`geometry/laplacian/`) | — | The weighted Laplacian (`-w` off the diagonal, PSD) and barycentric mass; `assemble_laplacian_system` over a membership-tagged triangle mesh and a stated edge weight, each row gathered on its own; `laplacian_form` the explicit free rows of `k_s L + k_b L M^-1 L + tau M` (`make_laplacian_form_pattern`, `state_laplacian_form_values`) over a free map whose out-of-range positions are fixed rows, `laplacian_is_anchored` the definiteness proof, `state_dirichlet_load` and `solve_dirichlet_form` the factor-once multi-field solve; the field, the fill's fairing and its chart residual read them |
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
| `polygon_intersections<Index, RealT, Int>` | 1, 2, or N tagged polygons + config | Intersection records whose ids are canonical point names — no coordinate table; self records when the config asks for `within` (implied for the one-form build) |
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
| `make_intersection_curves(poly0, poly1, config)` | `curves_buffer<Index, RealT, 3>` | Connected intersection curves between two meshes |
| `make_intersection_curves(range_of_forms, config)` | `curves_buffer` | All pairwise intersection curves for N meshes |
| `make_self_intersection_curves(polygons, config)` | `curves_buffer` | Self-intersection curves |
| `has_self_intersections(polygons)` | `bool` | Whether the mesh meets itself — the verdict of `tf::polygon_intersections`' one-form build: true exactly where that build would state a self record. It is that build's discovery tier, elections and all, stopped at the first record. Missing structures (tree, face membership, manifold edge link) are completed for the call, never required, so a bare mesh is a valid operand and a tagged form pays for none of them |
| `make_intersection_edges(si, faces)` | `blocked_buffer<Index, 2>` | The chords of every `(face, cut)` group as edge pairs; `tf::intersect::for_each_cut_group` finds the groups and `for_each_cut_chord` states one group's chords (`intersect/records/`), both shared with the iso module |

### The intersection run (`intersect_config`)

`{intersect_mode mode = primitives, double tolerance = 0.0}`, implicitly constructible from an `intersect_mode`.

`intersect_mode` carries the classifier and optional `within` request; `primitives | within` is canonical:

- `primitives` — each contact stated as what it is (EF/EE/VE/VF/VV); the
  default on every entry
- `sos` — every contact perturbed into a generic crossing, so all records are
  (edge, face) and none is ever coplanar. On an arrangement that means coplanar
  walls do not pool and the domains they would have separated stay joined —
  the caller's choice, stated in `arrangement_config`'s own doc

`within` asks for each form's own self-intersections. Its two halves — the
form's self records, and the resolution of crossings within one contour class —
are one request and never separable. A one-form build implies it.

Crossings between contours are resolved unconditionally, not on request:
between contour classes whenever a third tag exists to make such a pair, and
within one class whenever the run is `within`. That was always the behaviour of
every default; there is no flag left to turn it off with.

---

## 6. arrangement/

### The Arrangement

| Function / Structure | Return | Description |
|----------|--------|-------------|
| `make_arrangement_graph<Int>(operands, config)` | `arrangement_graph<Policy, Int>` | The build. Operands: a range of forms, one form (its self arrangement, `within` implied), or two forms of possibly different types. Completes any missing tree / face membership / manifold edge link in parallel |
| `arrangement_graph<Policy, Int>` | — | The arrangement of a set of forms, everything below classification: exposed triangle stream, cells, piece incidence, piece fences, coplanar stacks, `created_points()`, and `identities()` — the identity substrate both tiers state into |
| `arrangement_config` | — | `{intersect_config intersect, triangulation_type triangulation}`; implicitly constructible from either alone. Contour crossings are resolved unconditionally; under `sos` no contact is ever coplanar, so coplanar walls do not pool |

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
| `plane_world<Policy>` | The one carrier the plane arrangement's seam speaks; `make_plane_world(local_arrangement, identity_substrate)` borrows the production world and the substrate's local layer |
| `plane_arrangement<Index, Int>` | One triangulation per plane carrier: triangles, slot parents, corner subs, coplanar stacks, optional per-triangle cells. Owns no identity table: reads the local layer, states the plane layer through the substrate's verbs. A promoted face is seated by name (`seat_plane_promoted_faces`, for the refinement's promotion and the wave entrance alike): `promoted_plane_of()` names its plane, standing or founded. `build_diff` is the census round's value, carried across the renumbering |
| `seat_plane_promoted_faces(world, descriptors, face_frames, ...)` | THE SEATING LAW (`tf::intersect::graph::seat_named_planes`) against every plane the world holds: per face its plane and winding, per founded plane its frame and name |
| `make_plane_arrangement_cells(arrangement)` | `plane_arrangement_cells<Index>` — the 2-cells numbered densely across the planes |
| `make_plane_piece_incidence(arrangement, cells)` | `plane_piece_incidence<Index>` — piece <-> cell incidence from both sides |
| `make_plane_piece_fences(arrangement, world, incidence, cells)` | `plane_piece_fences` — per piece: `fan` and `crossable`, the one producer of both verdicts |

### Identity Substrate (`arrangement/identity/`, `tf::arrangement`)

| Symbol | Purpose |
|--------|---------|
| `identity_substrate<Index, Int>` | The one authority over an arrangement's identities: its layers `L` (`local_identity_tables`) and `P` (`plane_identity_tables`) are private, every table reads through it (`local_merges()`, `plane_merges()`, …) and every statement enters through a verb (`name_local_planes`, `state_local_names`, `close_local_classes`, `collapse_local_identities`, `mint_local_points`, `weld_local`, `seat_local_planes`, `seat_local_entrant_planes`; `mint_plane_classes`, `mint_plane_points` / `state_plane_point`, `weld_plane`, `seat_promoted_planes`, `seat_founded_planes`, `carry_plane`). `composed_target(source, id)` is the one closed-rewrite read (`P∘L`, `-1` for a key neither retires), the carve refusing an answer past its own currency (`reseat_refusal::name_foreign`); `local_target` / `plane_target` answer one layer; `plane_name(id)` / `plane_names()` answer every plane either tier holds |
| `composed_identity_point(identities, world, arrangement, get_mesh_point, tag, id)` | Where `(tag, id)` stands at `P∘L`: the one position law (`tf::intersect::graph::final_identity_point`) composed across both layers |
| `extent_shift<Index>` / `diff_extents<Index>` / `make_diff_extents(prior, n_prior_created, world)` | The one renumbering between a plane value and the next — created identities, planes and faces, each `[world extent \| promoted suffix]`: `forward`, `inverse`, `gap`, `lifted(n_flat)` |
| `carry_plane_identities(prior, created, ...)` (behind `identity_substrate::carry_plane`) / `compose_local_moves(...)` | A diff's plane layer carried across the shift, then the rows the statement moved in `L` composed into its evidence: forward, then append |
| `account_created_identities(identities, triangles, named, n_flat, n_created)` | Per created identity its account: `live`, `retired`, `vanished`, or the break (`emitted_dead`, `unaccounted`) |
| `assert_created_identities_are_accounted` / `assert_plane_layer_is_closed` / `assert_welds_are_lawful` | Verification checks: the completeness accounting and `P` closed over `L` on a published product, and the weld's law on a closed layer (one target per key, live targets, the layer's currency) |
| `mint_plane_classes(...)` | Behind `identity_substrate::mint_plane_classes`: a wave's births joined to the plane layer's classes, only a minting class persisting |
| `tf::intersect::graph::elect_identity_component(s)` | THE ELECTION (`intersect/graph/elect_identity_components.hpp`): rank `0` standing created, `1` fresh, `2` original leads the key; the smallest key speaks a coincidence, a standing identity survives, otherwise it mints, originals never survive; the caller states which candidates one coincidence joins |

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
`int32` (std), `int64` (std), `int128` (compiler-specific), `int256` (custom),
`int512` (custom)

`meta<Int>`: `T0` = self, `T1` = double width, `T2` = quadruple width, `T3` =
the construction rung

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
| `plane_support<Int>` | struct | The points a carrier stands on: `offer(p)` takes points in the caller's own order and keeps the first, the first distinct, and the first off their line, with the accepted cross as `normal`. The one producer of that scan; `make_supported_plane_frame` (the frame, its one reading), `tf::intersect::graph::plane_face_support` (face-shaped; `make_plane_face_frame` reads it for every tier that founds a plane on a face, so a collinear-corner face is a line carrier in all of them) and `make_face_plane` read it |
| `classify_segments(a0, a1, b0, b1, ...)` | `optional<pair<contacts>>` | Full segment-segment contact classification |

### Coordinate Conversion
- `pt_converter<Int, RealT, Dims>` — real → the `Int` lattice, scaled by the largest power of two within 99% of its max range
- `vertex_converter<Int, RealT, Dims>` — pt_converter + per-mesh globally unique vertex IDs
- `resolve_int_type<Int, Coord>` — the standing law: `tf::none_t` becomes int32
  for float, int64 for double, `Coord` itself for an integral `Coord`. Every
  factory (`make_pt_converter`, `make_vertex_converter`,
  `tf::make_exact_coordinate_converter`, `tf::make_exact_points`) leaves both
  `Int` and `RealT` unstated by default and reads them off its input
- `input_lattice<Index, RealT, Int>` — where every original vertex of every
  operand stands on the lattice, built once over the union of the operands:
  the shared converter, the flat vertex space that names `(tag, id)` by one
  integer, and — when a tolerance is given — the sparse table of the
  vertices the door (`exact/door/`) moved (`moved_points`, `moved_tickets`)
  and its `census()`, plus the edge landings (`edge_names`). A band does
  two things: faces it could have made of one plane POOL
  (`exact/door/pool/pool_input_planes.hpp`: every exact plane quantized,
  direction onto a fixed chart and intercept onto the band, equal keys one
  pool across all forms, a key held by one plane pooling nothing) onto one
  committed plane the lattice is dense on, which their vertices ride — a
  vertex that cannot stand on every pool around it stays put and every pool
  at it releases its faces there — and a
  RIM vertex — on an edge one face holds — within the band of another
  form's face lands on that face's nearest feature (`find_rim_targets`):
  the corner's point, the lattice point nearest its projection onto an
  edge plus a `{vertex, face, slot}` name the intersection build states as
  a vertex-on-edge record (`state_rim_edge_names`), or a lattice point on
  the face's committed plane (`commit_target_planes`); `place_door_vertices`
  places every vertex once. A rim vertex exactly on the face, or whose rim
  edges already cross the other form's edges, is kept; interior vertices
  never move. A tolerance of zero is the identity: no pool, no table, no face
  read, the plain converter
- `input_lattice_reader<Index, RealT, Int, ApplyToForm>` — the one reader of an
  original vertex's position, bound to the caller's own forms: `(tag, id)` or a
  flat id answers its moved row when the band moved it and the converter over
  the input's own coordinate otherwise; the real-valued export of an unmoved
  vertex is the input's own coordinate
- `input_lattice_tail_reader<Index, RealT, Int, ApplyToForm>` — the reader of
  a record face's corners, made by `input_lattice::tail_reader`: an id past its
  form's own points is a seat answered from the caller's per-tag tail, every
  other id the one reader's answer; a read through a form's own faces never
  names a seat and takes the plain reader

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
| `read_nifti<T>(path)` | `nifti_file<T>` | NIfTI-1 volume, `.nii` or `.nii.gz` (vendored miniz): `volume_buffer<T, float, 3>` + the frame the file's affine states + `posed` + `nifti_status`. `T` defaults to float; the file's dtype converts through one cast, `scl` scaling applied. The read splits the file's one affine canonically — spacing and axis-aligned translation onto the grid, orientation into the frame |
| `read_nifti_header(path)` | `nifti_header_info` | The file's facts (dtype, dims, spacing, posed, scl) without its samples; a gzipped file inflates only its head |
| `write_nifti(vol[, frame], path)` | `bool` | NIfTI-1 out, gzip by `.gz` extension; samples in their own type, the grid and frame composed into the sform |

Memory buffer variants accept `range<const char*, dynamic_size>` for in-memory parsing.

## 12. volume/

A dense scalar field on a regular axis-aligned grid: its carrier, the
generators that fill one, the CSG that combines two, the plane it slices onto,
and the geometry its level set names. Umbrella `trueform/volume.hpp`; the
machinery is `tf::volume_detail` under `volume/impl/`. A grid of three axes is
a voxel grid, one of two a pixel grid, and a SLICE of a 3D field IS a 2D one —
nothing but the axis count differs.

### The carrier

| Type | Purpose |
|------|---------|
| `volume<Policy>` | The samples range wrapped in the grid that names them, a `tf::form` like points and polygons. A volume is a sampled function, so the type answers TWO questions — `sample_type` (the codomain: what was measured) and `coordinate_type` (the domain: where samples stand) — the same by default and stated or factory-deduced apart (int16 CT samples on a float millimetre grid). The AXIS COUNT is deduced through `tf::coordinate_dims_v` from the grid policy (`volume_detail::grid<Range, Coord, Dims>`, `volume/impl/grid.hpp`), never stated by a reader. `dims`, `spacing`, `origin`, `voxel_count`, `linear_index`, `operator()`, `point_at<Real>` — each taking one integer per axis or the `std::array` of them; built by `make_volume(samples, dims, spacing, origin)`. World placement is a tagged frame (`vol \| tf::tag(frame)`): the geometry-emitting entries emit through it — a reflecting frame keeps the outward winding — and positional inputs stay local-space |
| `volume_buffer<T, Coord = T, Dims = 3>` | Owns the samples in `T` — any arithmetic scalar, int16/uint16/uint8 included — on a `Coord` grid, payload first and coordinates second as `polygons_buffer` orders them; `volume()` yields the view. `make_volume_buffer<OCT>(view)` copies one: OCT stated decides both types, unstated preserves both |

Samples are stored with the first axis varying fastest:
`x + dims[0] * (y + dims[1] * z)`. An entry CONSTRAINS on the axis count
through the trait, never through a parameter — `make_isosurface` asserts three,
`make_isocontours`' grid overload two, `make_mesh_sdf` three (the parity
sign is a closed surface's), while `make_boolean` and `make_sphere_sdf` hold at
either.

### The type a call decides in

An entry that takes a volume resolves `OutputCoordinateType` through
`tf::resolved_output_real_t` against that volume's COORDINATE type, so an
unstated request is the grid's — which is the samples' type in the
same-by-default case, and the floating domain type for an integer-sampled
field. `make_mesh_sdf` takes no volume and resolves against the MESH's
coordinate type; `make_sphere_sdf` has no OCT slot at all — its `T` is the
field's type outright. An entry that also names an index type takes `Index`
FIRST and `OCT` second (`make_isosurface<Index, OCT>`,
`make_isocontours<Index, OCT>`), as core's own geometry entries do; one
with no index slot takes `OCT` alone. That resolved type is the ONE type the
call decides and emits in, and `tf::volume_detail::field_value` is the one cast
between a stored sample and it — which is what keeps the classifier and the
crossing that follows it on the same number, and a crossing parameter inside
`[0, 1]`.
`volume_detail::field_samples` is its pointer-carrying form, held by the passes
the deciding type has to travel through.

| Function | Return | Description |
|----------|--------|-------------|
| `make_isosurface<Index, OCT>(vol[, iso][, config])` | `polygons_buffer<Index, RealOut, 3, 3>` | The level set as a welded indexed triangle mesh; a corner is inside when `sample < iso`, so an SDF winds outward |
| `isosurface_config` | — | `{isosurface_method method, bool refine, double stabilizer}`, implicitly constructible from an `isosurface_method` |
| `isosurface_method` | — | `flying_edges` (a vertex per crossing grid edge; any field) or `dual_contouring` (a vertex per surface component of a cell, fitted to the crossings; a distance-like field) |
| `make_sphere_sdf<T>(dims, spacing, origin, center, radius)` | `volume_buffer<T, T, Dims>` | The analytic sphere field; Dims-generic, a 2D `dims` stating a disc |
| `make_mesh_sdf<OCT>(polygons, dims, spacing, origin[, config])` | `volume_buffer<RealOut>` | Nearest-distance magnitude, sign by exact crossing parity on the lattice (negative inside by winding); closed surface, needs only the tree — a tagged one is used, a missing one is built for the call — in the frame the form states. `mesh_sdf_config{mesh_sdf_mode::banded, band}` measures a band and sweeps the far field |
| `make_boolean<OCT>(a, b, op)` | `volume_buffer<RealOut, RealOut, Dims>` | The library's `make_boolean`, type-distinguished on volume carriers and `volume_boolean_op{union_, intersection, difference}`. The combine happens on one shared grid: matched grids AND matched poses combine samplewise (a POSED call returns `(buffer, pose)` — the shared pose kept); anything else resamples onto the world-space union of the domains at the finer LOCAL spacing through `make_resampled_volume` and returns identity pose. Out of an operand's domain is outside its solid: every operand reads the far-outside sentinel past its own box. Pose equality is exact matrix equality in the deciding type. An empty operand is the empty set and the algebra answers for it. Refuses unsigned samples and non-floating deciding types. Dims-generic: the algebra is the field's |
| `make_isocontours<Index, OCT>(grid_2d, isovalue-or-range)` | `curves_buffer<Index, RealOut, 2>` | The level set of a 2D volume, connected into polylines — core's isocontours product, one dimension down |
| `make_isocontours<Index, OCT>(vol, plane_origin, u, v, dims2, spacing2, isovalues)` | `curves_buffer<Index, RealOut, 3>` | The slice-then-contour composition plus the lift back into the volume's frame, and the one producer of that lifted product |
| `make_resampled_volume<OCT>(vol, dims, spacing, origin)` | `volume_buffer<RealOut, RealOut, Dims>` | The one regrid: multilinear at each target node, clamp-to-edge untagged; a POSED volume resamples through its frame's inverse with out-of-domain nodes at the far-outside sentinel. Two reads, two contracts: `make_boolean`'s general path — mismatched grids or poses — consumes the POSED one for every operand, an untagged one through the identity |
| `make_volume_slice<OCT>(vol, plane_origin, u, v, dims2, spacing2)` | `volume_buffer<RealOut, RealOut, 2>` | The 3D field multilinearly resampled onto an oriented plane — a volume one dimension down; nodes outside the box take a sentinel above the field maximum |

### The extractors (`volume/impl/`, `tf::volume_detail`)

| Symbol | Purpose |
|--------|---------|
| `classify_x_edges<Real>(vol, iso, ...)` | The x-edge classification both extractors stand on: a 2-bit case per edge, each row's crossing count and trim |
| `flying_edges<Index, Real>(vol, iso)` | Flying Edges 3D: classify, count, one serial row prefix, generate — the welded mesh directly |
| `flying_edges_2d<Index>(grid_2d, iso, points, segments)` | Its planar sibling, behind `make_isocontours`; the product is welded points plus the segments over them |
| `dual_contouring<Index, Real>` | The DC builder on the same chassis: one vertex per surface component, arcs given identity so the surface is manifold by construction, and an optional refit of feature vertices to the planes the samples state. `record_polygons` / `record_refit_provenance` / `record_pass_times` are requests a consumer makes before the build |
| `marching_cubes<Index, Real>(vol, iso)` | The reference baseline, retained; same surface, orientation and output shape |

---

## 13. linalg/

Sparse symmetric positive definite solves. Umbrella `trueform/linalg.hpp`;
the steps are `tf::linalg` under `linalg/cholesky/`. Native only.

| Symbol | Purpose |
|--------|---------|
| `symmetric_lower<Index>` | A symmetric matrix by its lower triangle: rows of ascending columns `j <= i`, diagonal last; an entry's position in `columns` is its ticket |
| `cholesky_pattern<Index>` | The factor's structure, read once per pattern: the elimination order and its inverse, the permuted lower carrier with its `lower_slot` tickets, the supernodes with their rows, tree and value offsets, and the schedule (`block_offsets`, `frontier_first()`) |
| `cholesky_factor` | One value set's factor: one dense block per supernode where the pattern places it |
| `cholesky_scratch<Index>` | The caller-owned workspace every step walks on |
| `analyze_cholesky(matrix, scratch, pattern)` | Fills the pattern from the matrix's pattern alone, under approximate minimum degree composed with a postorder; `false` where the index type cannot name the factor's values |
| `factor_cholesky(pattern, values, scratch, factor)` | One value set, multifrontal over the supernodes with dense tiled kernels, the subtree blocks in parallel and then the frontier; refuses a pivot that is not positive or not finite; its determinism is stated in the header |
| `solve_cholesky(pattern, factor, rhs, r, scratch)` | `r` interleaved right-hand sides in place, one pass over the factor per direction: each supernode's lanes gathered into one run and its block walked in storage order, the backward solve taking the frontier and then the subtree blocks in parallel; its determinism is stated in the header |
| `linalg::refine_once` | One refinement step, accepted where every row's componentwise backward error is within `cholesky_residual_tolerance` |
| `linalg::analyze_cholesky_order` | The pattern under a stated elimination order, which `analyze_cholesky` calls under minimum degree |
| `linalg::make_symmetric_adjacency` / `order_minimum_degree` / `make_elimination_tree` / `postorder_tree` / `make_permuted_lower` / `count_cholesky_columns` / `make_supernodes` / `make_supernode_rows` / `partition_elimination_tree` / `factor_supernode_run` / `factor_supernode` / `extend_add_update` / `factor_front_columns` / `factor_diagonal_tile` / `solve_panel_rows` / `subtract_tile_product` / `solve_lower` / `solve_upper` / `eliminate_column` / `back_substitute_column` / `column_dot` | The steps |

---

## 14. csg/

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
| `make_csg_open_fragments(graph[, domain_config])` | `(fragments, tags)` | Per form, the live triangles of every component whose two coarse sides are one domain — exactly what `make_csg_domains` leaves out under the same `config` — wound as stored, so cells and fragments together name every live face. Same `return_source_ids` / `return_index_map` overload set; the index map's `inclusion` holds no row, a fragment bounding no domain. A staged hand-off passes the fragments on beside the cells, each keeping its form's sheet or volume declaration |
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

---

## 15. fill/

Hole filling. Umbrella `trueform/fill.hpp`; the machinery is `tf::fill`
under `fill/holes/`.

### Public Surface

| Function / Structure | Return | Description |
|----------|--------|-------------|
| `fill_holes<Int>(polygons, rims[, config])` | `hole_fill_result<Index, RealT>` | One group per rim of `tf::make_boundary_rims`; the form needs face membership (tag it or it is built). A carrying face must be a triangle; a tagged frame does not enter |
| `make_filled_mesh<Index>(polygons, result)` | `polygons_buffer` | The replay: faces no split reached copied at their own arity, plan faces replaced by their plans, patches appended; points = input then the minted lane read flat. A triangle input stays triangles, anything else comes back dynamic |
| `hole_fill_config` | — | `{double min_quality = 0.3, bool fairing = true, bool split_rim = true}` |
| `hole_fill_result<Index, RealT>` | — | Per group: `group_offsets`, `triangles` / `minted_points` (offset blocks), `status`, `charted`, `refined`, `faired`, `seam_max_angle`, `offending` (`(rim, edge)`, `(-1, -1)` none). Global: `splits` (canonical, `hole_split{face, v0 < v1, point, parameter}` over `hole_split_scale`), `plan_faces` ascending + `plan_triangles` |
| `hole_fill_status` | — | `filled`, `refused_open`, `refused_invalid_rim`, `refused_host_face`, `refused_protection_depth`, `refused_table`, `refused_resource`, `refused_rim_split` |
| `hole_chart_status` | — | `not_attempted`, `admitted`, `rim_plane`, then the refusals `no_area`, `steep`, `folded`, `collar_arity`, `uncovered`, `unsampled`, `unsolved`, `forbidden_chord`, `resource`, `out_of_lattice` |
| `hole_refine_status` / `hole_fair_status` | — | `not_attempted`, `floor_met`, `stalled`, `budget_exhausted` / `not_attempted`, `faired`, `fairing_off`, `fairing_refused` |

A corner below the input's point count is an input vertex; at or above it,
it names `minted_points` read flat in group order. A refused group
publishes nothing and names its offending rim edge.

### The Ladder

One rim is one group, validated by preflight and then handed down: TINY
(at most five vertices, exhaustive enumeration) -> CHART, continue-or-lid
(the hole as a height field over the rim's Newell plane: a trend — one
sphere or quadratic, a consensus of subsample fits on a large collar —
plus a biharmonic residual on the graded samples' triangulation joined to
ring 1; under any chart refusal a rim lying exactly in one plane is lidded
in it as `rim_plane`) -> TABLE (the rim's sites tetrahedralized, a
two-stage table states the least-bottleneck then least-area patch, refined
toward `min_quality` and faired). The chart's verdict is reported whichever
tier held the rim. `split_rim` lets a tier split an original rim edge where
its patch needs one; every split is a dyadic record in the one canonical
table, its carrying faces get plans, and `make_filled_mesh` replays both,
so the stitched mesh is watertight. Without it no original edge is touched
and a rim fillable only by a split refuses `refused_rim_split`.

THE QUANTUM LAW: the chart decides on the mesh's own lattice — the quantum
the fill's input converter states (`tf::fill::hole_lattice_unit`), one for
the whole fill as one lattice serves an arrangement. Its headroom law is
stated once in `fill/holes/hole_chart.hpp`; a collar past the lattice
refuses `out_of_lattice`.

### The Tiers (`fill/holes/`, `tf::fill`)

| Symbol | Purpose |
|--------|---------|
| `fill_hole_rims` | The driver: overlap rejection, per-rim tiers in parallel, canonical split merge, face plans, fairing |
| `preflight_hole_rim` / `state_hole_rim_cycle` | Rim validation and rooting |
| `fill_tiny_hole` | Exhaustive enumeration, at most five vertices |
| `fill_chart_hole` | The chart tier: `make_hole_rim_collar`, `hole_chart`, `fit_hole_trend`, `sample_hole_chart`, `solve_hole_chart_residual`, `place_hole_chart_samples`, `emit_hole_chart_patch`; `make_hole_lid_patch` for `rim_plane` |
| `fill_table_hole` | The table tier: Delaunay sites, `protect_hole_rim`, `solve_hole_table`, `refine_hole_patch` |
| `make_hole_face_plans` | The carrying faces' plans over the surviving split set |
| `fair_hole_patches` | The fairing stencil, solved over `tf::geometry`'s laplacian system (`geometry/laplacian/`) |
| `hole_patch_seam_angle` | The seam lane |

---

## 15. tetmesh/

Constrained Delaunay tetrahedralization of closed surfaces. Umbrella
`trueform/tetmesh.hpp`; the machinery is `tf::tetmesh` under
`tetmesh/entrance/`, `tetmesh/carve/` and `tetmesh/mesh/`, over the
constraint wave in `topology/cdt/dt3/recovery/`.

### Public Surface

| Function / Structure | Return | Description |
|----------|--------|-------------|
| `make_tet_mesh<Int, OCT>(polygons)` | `(tetrahedra_buffer<Index, RealOut>, offset_block_buffer<Index, std::array<Index, 3>>, offset_block_buffer<Index, Index>, tet_mesh_report)` | The solid the surface winds: entrance, facet merging, Delaunay build, constraint wave, carving by winding, the positively wound cells kept. Coplanar triangles that meet wound alike are one facet (exact plane names, `tf::intersect::graph::name_plane_carriers`' law), recovered as one region whose interior is triangulated freely. Block `k` of the second holds the output triangles covering facet `k`, wound as it is; block `f` of the third the facets input face `f` lies in, ascending. One instantiation per input type: the cheapest rung that holds its precision (int32 lattice in int64 sites, 40 bits, for float; int64 in int64, 63, for double), the input quantized once. A constraint its frame splits too coarsely deepens past it site by site — a segment its own frame, a face the point it places — each such site standing below the frame as deep as the rung's deep lattice admits (float 23 bits on int64, double 20 on int64 in int128), and every output point is a site's exact position mapped back. Minted sites are bounded at `tet_execution_tuning::site_budget_ratio` per entrance site. `Int`, when stated, restricts the rung to that lattice; `OCT` is the floating-point output |
| `tet_mesh_status` | — | `meshed`, `refused` (some constraint refused, for corners, depth, the site budget or the face budget; a mesh only if the surface still winds consistently), `unwound` (open or inconsistently wound), `rank_deficient`, `capacity` |
| `tet_mesh_report` | — | `status`; `refused_segments`, `refused_depth` (segments and faces whose frames ran out at the deep lattice's cap), `refused_budget`, `refused_faces`; the lattice that ran: `rung`, `storage_bits`, `input_bits`, `frame_bits`, and the sites below its frame, `deep_sites` and `deepest`; the measured `corner_bits_needed` and `min_angle`, reported, never chosen by; `flat` (cells coplanar) and `inverted` (cells an export narrower than the lattice reads negative); and the wave's `census` |

### The Tiers (`tf::tetmesh`)

| Symbol | Purpose |
|--------|---------|
| `measure_tet_entrance_need(polygons)` | `tet_entrance_need`: the exact lattice's unit, offset and width, and the corner estimate (`measure_tet_corner_need`) |
| `make_tet_entrance<Index, Int, Coord>(polygons, need, exact)` | The input onto the exact lattice through the integers, or at its own precision rounded once; its self-arrangement cuts crossings at their canonical identities; `origins` names each site's input point; sites stored at `2^frame_bits` of the lattice |
| `tet_entrance_rung_types<R>` / `tet_entrance_budget<Int, Coord>` | The two rungs, int32 in int64 and int64 in int64, and each rung's width |
| `tet_entrance_budget<Int, Coord>` | `width` the site width the predicates admit (`tf::topology::cdt::dt3::tet_site_width`); `lattice_bits = min(converter_bits, width − 10)`: the house converter's width less ten frame bits — float 30 + 10 on int32, double 53 + 10 on int64 |
| `tf::topology::cdt::dt3::tet_deep_lattice<Int, Coord>` | Where a site below the frame is read: the lattice and coordinate an aligned decision runs on and `cap`, the depth it admits — 23 for int32 sites in int64 (read on int64), 20 for int64 in int64 (on int64 in int128). A site's position is its floor plus a canonical residual at its depth (`tet_position`), minted once (`tet_minted_position`) and read through one accessor (`tet_site_lanes`); a decision reads every corner at the deepest depth among them (`tet_with_sites`), on the floors while every site stands on the frame |
| `carve_tet_complex(owner, faces, owned, carving)` | Every finite cell's winding number: regions joined across unprotected facets, crossings stated by the covering faces' orientation, walked from the unbounded region; `consistent` false when crossings disagree |
| `emit_tet_mesh` / `emit_tet_face_children` | The kept cells and their points back in the input's coordinates; the covering triangles per input face |
