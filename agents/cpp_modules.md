# C++ Module-by-Module API Reference

All modules follow the same conventions: `snake_case` naming, `build()` pattern for stateful structures, `make_*()` free functions for convenience construction, `tf::none_t` for type deduction, parallel-by-default via TBB.

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
| `connect_edges_to_paths(edges)` | edges | `offset_block_buffer<Index, Index>` | Assemble edges into continuous paths |
| `boundary_paths(polygons)` | polygons | `offset_block_buffer<Index, Index>` | Boundary edges assembled into paths |
| `find_eulerian_paths(edges, link, ...)` | edges + link | offsets + edge IDs (output params) | Hierholzer's edge-disjoint path decomposition |
| `label_connected_components(labels, applier)` | label buffer + neighbor callback | `int` (n_components) | Parallel union-find component labeling |
| `make_manifold_edge_connected_component_labels(polygons)` | polygons | `connected_component_labels<Index>` | Component labeling via manifold edge adjacency |
| `orient_faces_consistently(polygons)` | polygons | void (modifies in place) | Consistent face orientation via weighted voting |

### Planar Graph Processing

| Structure | Build Input | Purpose |
|-----------|-------------|---------|
| `planar_graph_regions<Index, Int>` | directed edges + 2D points | Extract minimal closed regions by polar angle walk |
| `face_hole_relations<Index, Int>` | faces + holes + points | Parent-child nesting between faces and holes |
| `face_split_by_edges<Index, Int>` | face loop + edges + points | Subdivide a face by interior edges |
| `hole_patcher<Index, Int>` | outer loop + holes + points | Bridge holes into outer boundary |
| `planar_embedding<Index, Int>` | directed edges + points | High-level: regions + hole relations |

### Triangulation

`ear_cutter<Index, Int>` — Exact ear-clipping with collinear run handling, hole support, z-order optimization. `build(ids, points)` returns `bool`. Access via `faces()` and `indices_buffer()`.

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
| `neighbor_search(form, query, radius_sq)` | + radius | `tree_metric_info` | Within radius |
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
| `fit_similarity_alignment(X, Y)` | `transformation` | Procrustes with uniform scale |
| `fit_knn_alignment(X, Y, state, config)` | `transformation` | Gaussian-weighted k-NN soft correspondence |
| `fit_obb_alignment(X, Y, sample_size)` | `transformation` | OBB alignment with ambiguity resolution |
| `fit_icp_alignment(X, Y, state, config)` | `transformation` | Iterative closest point with convergence |

### Processing

| Function | Return | Description |
|----------|--------|-------------|
| `triangulated(polygons)` | `polygons_buffer<..., 3>` | Ear-clipping triangulation |
| `laplacian_smoothed(points, iters, lambda)` | `points_buffer` | Laplacian smoothing |
| `taubin_smoothed(points, iters, lambda, kpb)` | `points_buffer` | Volume-preserving smoothing |
| `ensure_positive_orientation(polygons)` | void (in-place) | Outward-facing normals |
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
| `intersections_between_polygons<Index, RealT, Int>` | 2 or N tagged polygons + mode | Intersection records in int32 coords |
| `intersections_within_polygons<Index, RealT, Int>` | 1 polygon set + mode | Self-intersection records |
| `intersection_graph<Index, Int>` | intersections + face accessor + point accessor | Loops, edges, points per face |

### High-Level Functions

| Function | Return | Description |
|----------|--------|-------------|
| `make_intersection_curves(poly0, poly1, mode)` | `curves_buffer<Index, RealT, 3>` | Connected intersection curves between two meshes |
| `make_intersection_curves(range_of_forms, mode)` | `curves_buffer` | All pairwise intersection curves for N meshes |
| `make_self_intersection_curves(polygons, mode)` | `curves_buffer` | Self-intersection curves |

### Intersection Modes (`intersect_mode`, bitwise)
- `sos` — SoS fan triangulation (all edge-face records)
- `primitives` — Full 5-type classification (EF/EE/VE/VF/VV)
- `resolve_crossing_contours` — Resolve inter-contour crossings
- `resolve_self_crossing_contours` — Resolve intra-contour crossings

---

## 6. cut/

### Boolean Operations

| Function | Return | Description |
|----------|--------|-------------|
| `make_boolean(poly0, poly1, op)` | `(mesh, labels, face_labels)` | Union/intersection/difference |
| `make_boolean(..., return_curves)` | `+ curves_buffer` | Also returns intersection curves |
| `make_boolean_pair(poly0, poly1, op)` | `(left_mesh, right_mesh)` | Both halves with open boundaries |
| `make_mesh_arrangements(poly0, poly1)` | `(mesh, tag_labels, face_labels)` | Decompose into classified regions |
| `make_mesh_arrangements(range_of_forms)` | `(mesh, tag_labels, face_labels)` | N-mesh arrangements |

### Boolean Operations (`boolean_op`)
- `merge` (union), `intersection`, `left_difference` (A-B), `right_difference` (B-A)

### Face Classification (`arrangement_class`)
- `inside`, `outside`, `aligned_boundary`, `opposing_boundary`

### Infrastructure

| Structure | Purpose |
|-----------|---------|
| `face_cuts<Index, Int>` | Subdivide all intersected faces |
| `face_cutter<Index, Int>` | Subdivide single face |
| `cut_graph<Index>` | Per-edge connectivity from face cuts |

---

## 7. exact/

### Integer Types
`int32` (std), `int64` (std), `int128` (compiler-specific), `int256` (custom)

`meta<Int>`: `T0` = self, `T1` = double width, `T2` = quadruple width

### Predicates

| Function | Return | Description |
|----------|--------|-------------|
| `orient3d_sos(vertices[4])` | `bool` | 3D orientation with SoS (never zero) |
| `orient3d_sign(vertices[4])` | `int` (-1/0/+1) | 3D orientation exact sign |
| `orient3d_value(p0, p1, p2, p3)` | `T2` | Signed volume × 6, exact |
| `orient2d(p0, p1, p2)` | `T2` | Signed area × 2, exact |
| `orient2d_sos(v0, v1, v2, ax0, ax1)` | `bool` | 2D orientation with SoS |
| `signed_area_2x(polygon)` | `T2` | Polygon signed area × 2 |
| `signed_area_sign(polygon)` | `int` | +1 (CCW), -1 (CW), 0 |
| `projection_axes(p0, p1, p2)` | `pair<int,int>` | Best 2D projection axes for 3D triangle |
| `classify_segments(a0, a1, b0, b1, ...)` | `optional<pair<contacts>>` | Full segment-segment contact classification |

### Coordinate Conversion
- `pt_converter<Int, RealT, Dims>` — float → int32 (99% of INT_MAX range)
- `vertex_converter<Int, RealT, Dims>` — pt_converter + per-mesh globally unique vertex IDs

---

## 8. clean/

| Function | Input | Return | Description |
|----------|-------|--------|-------------|
| `cleaned(polygons)` | polygons | `polygons_buffer` | Remove degenerate faces + duplicate vertices |
| `cleaned(polygons, tolerance)` | + tolerance | `polygons_buffer` | Tolerance-based deduplication |
| `cleaned(segments)` | segments | `segments_buffer` | Remove zero-length edges |
| `cleaned(points)` | points | `points_buffer` | Deduplicate points |
| `cleaned(curves)` | curves | `curves_buffer` | Deduplicate + reconnect paths |

All have `return_index_map` variants returning `(buffer, face_map, point_map)`.

Low-level: `make_clean_index_map(geometry)` returns maps without applying them.

---

## 9. reindex/

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

## 10. io/

| Function | Return | Description |
|----------|--------|-------------|
| `read_stl(path)` | `polygons_buffer<int, float, 3, 3>` | Binary/ASCII STL with auto-deduplication |
| `write_stl(polygons, path)` | `bool` | Binary STL (<500MB buffered, >500MB streamed) |
| `read_obj(path)` | `polygons_buffer<int, float, 3, dynamic_size>` | ASCII OBJ (1-based → 0-based indices) |
| `read_obj<3>(path)` | `polygons_buffer<int, float, 3, 3>` | OBJ with fixed triangle faces |
| `write_obj(polygons, path)` | `bool` | ASCII OBJ (parallel two-pass write) |

Memory buffer variants accept `range<const char*, dynamic_size>` for in-memory parsing.
