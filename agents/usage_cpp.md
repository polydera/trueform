# Using Trueform from C++

> **Caller-facing guide.** Read this when writing examples, applications,
> bindings, or public documentation that consumes Trueform. It explains how to
> compose the public API. It is not an implementation-performance guide; for
> core algorithm work read `cpp_performance_philosophy.md` and
> `cpp_execution_patterns.md`.

The full public reference lives in `docs/content/cpp/2.modules/`. In particular,
`01.core.md` defines the type, view, buffer, and policy vocabulary used by every
other module.

## 1. The caller model

Trueform is built around five ideas:

1. Geometry is expressed with typed primitives, not anonymous numeric arrays.
2. Caller-owned storage is exposed as zero-copy semantic views.
3. Owning buffers appear at materialization and result boundaries.
4. Policies attach reusable capabilities to views.
5. Algorithms consume the richest capabilities already present and build
   missing ones only for convenience.

The normal high-performance calling pattern is:

```text
own stable geometry storage
-> create points/faces/polygons views
-> precompute reusable structures in dependency order
-> tag every structure the planned pipeline consumes
-> create lazy transformed instances when needed
-> call algorithms on the tagged forms
-> retain returned buffers, labels, and index maps
```

Plain one-shot calls are valid. Repeated work should not repeatedly ask the
convenience layer to rediscover the same tree, topology, normals, or half-edges.

## 2. Includes and public namespace

Include everything:

```cpp
#include <trueform/trueform.hpp>
```

Or include module umbrellas:

```cpp
#include <trueform/core.hpp>
#include <trueform/spatial.hpp>
#include <trueform/topology.hpp>
#include <trueform/arrangement.hpp>
```

Most public symbols live directly in `tf::`. The intentionally public nested
namespaces are `tf::exact` for arrangement/intersection integer types and
`tf::csg` for
CSG expressions. Do not build application code on implementation namespaces.

## 3. Use Trueform primitives for geometry

Do not represent semantic geometry as anonymous C arrays, `std::array`, or
tuples throughout application code. Use Trueform's owning primitive types:

```cpp
tf::point<float, 3> position{1.f, 2.f, 3.f};
tf::vector<float, 3> direction{1.f, 0.f, 0.f};
tf::unit_vector<float, 3> normal{direction}; // normalizes

auto segment = tf::make_segment_between_points(position0, position1);
auto ray = tf::make_ray(position, direction);
auto plane = tf::make_plane(normal, position);
auto box = tf::make_aabb(min_point, max_point);
```

Factories preserve semantic type while deducing scalar type and dimension:

```cpp
auto point = tf::make_point(1.f, 2.f, 3.f);
auto vector = tf::make_vector(1.f, 0.f, 0.f);
auto unit = tf::make_unit_vector(vector);
auto line = tf::make_line_between_points(point0, point1);
auto polygon = tf::make_polygon(face_ids, points);
```

The distinction is useful, not cosmetic:

- point minus point produces a vector;
- point plus vector produces a point;
- point plus point and point-times-scalar are intentionally unavailable;
- unit vectors normalize on construction unless `tf::unsafe` explicitly states
  that the caller already guarantees unit length;
- transformations treat points, vectors, normals, and planes according to their
  geometry rather than as interchangeable numeric arrays.

Structured bindings are available when coordinates or members are needed:

```cpp
auto [x, y, z] = point;
auto [a, b] = segment;
auto [p0, p1, p2] = triangle;
```

### Views at an interop boundary

Raw arrays are appropriate as external storage. Wrap them immediately in a
typed primitive view instead of passing raw coordinate triples through the
algorithm:

```cpp
float coordinates[3] = {1.f, 2.f, 3.f};
auto point = tf::make_point_view(coordinates);

float *external = get_external_point();
auto point3 = tf::make_point_view<3>(external);
```

The array owns memory; `point_view` supplies geometry. Keep the array alive and
stable for the lifetime of the view. Copy into `tf::point<T, Dims>` when the
primitive must own its value independently.

For generic code, accept the appropriate `*_like` concept/type shape already
used by neighboring Trueform APIs rather than falling back to `T*` plus a
dimension argument.

## 4. Storage first, semantic views second

### External storage

Views do not copy their backing data:

```cpp
std::vector<float> coordinates = {
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
};
std::vector<int> indices = {0, 1, 2};

auto points = tf::make_points<3>(coordinates);
auto faces = tf::make_faces<3>(indices);
auto polygons = tf::make_polygons(faces, points);
```

`points`, `faces`, and `polygons` borrow the vectors. Keep the vectors alive and
do not reallocate them while those views are in use.

The same applies to views built from pointers:

```cpp
auto points = tf::make_points<3>(tf::make_range(point_ptr, n_points * 3));
auto faces = tf::make_faces<3>(tf::make_range(face_ptr, n_faces * 3));
auto polygons = tf::make_polygons(faces, points);
```

### Owning buffers

Trueform result/storage types own flat memory and expose semantic views:

```cpp
tf::polygons_buffer<int, float, 3, 3> mesh;

auto polygons = mesh.polygons(); // borrowed combined view
auto faces = mesh.faces();       // borrowed connectivity view
auto points = mesh.points();     // borrowed point view
```

Common duals are:

| Owning type | Borrowed semantic view |
|---|---|
| `points_buffer` | `points()` |
| `segments_buffer` | `segments()`, `edges()`, `points()` |
| `polygons_buffer` | `polygons()`, `faces()`, `points()` |
| `curves_buffer` | `curves()`, `paths()`, `points()` |
| `blocked_buffer` | fixed-size block range |
| `offset_block_buffer` | offset-block range |

Keep the owning result alive while using any accessor view. Moving or
reallocating the owner may invalidate previously obtained views.

### Allocate versus append

`tf::buffer::allocate(n)` creates `n` logical, uninitialized elements. Fill them
by index or with an algorithm. Use `reserve(n)` when the next operation is
`push_back`/`emplace_back`.

```cpp
tf::points_buffer<float, 3> points;
points.reserve(1000);
points.emplace_back(1.f, 2.f, 3.f);

tf::buffer<float> values;
values.allocate(1000);
tf::parallel_fill(values, 0.f);
```

`release()` transfers a Trueform allocation. Free it with the matching
`tf::deallocate<T>`, never `delete[]`.

## 5. Preserve the carrier and its static shape

Fixed topology uses fixed blocked views:

```cpp
auto triangle_faces = tf::make_faces<3>(triangle_ids);
auto quad_faces = tf::make_faces<4>(quad_ids);
```

Variable topology uses offsets plus packed IDs:

```cpp
tf::buffer<int> offsets = {0, 3, 7};
tf::buffer<int> data = {0, 1, 2, 3, 4, 5, 6};
auto dynamic_faces = tf::make_faces(offsets, data);
```

Static arity propagates through Trueform ranges and selects specialized
implementations. Do not erase triangle/fixed-block information into a dynamic
carrier unless mixed arity is genuinely required.

`points`, `segments`, and `polygons` are **forms**: their elements are
independently queryable by spatial structures. Vectors, unit vectors, and curves
are semantic ranges but are not spatial forms.

## 6. Topology is identity, not coordinate equality

Indexed polygons are connected only when faces reference the same point IDs.
Two equal coordinates stored under different IDs are topologically distinct.

```cpp
auto soup = tf::make_polygons(tf::make_blocked_range<3>(raw_triangle_points));
auto indexed = tf::cleaned(soup); // materializes shared point identities
```

Clean a soup when coordinate duplicates are intended to become shared vertices.
Do not clean an already indexed mesh merely by habit: seams and deliberately
duplicated vertices can carry real meaning.

Returned topology is usually keyed by an existing carrier:

- face membership: block `vertex_id -> face_ids`;
- vertex link: block `vertex_id -> neighbor_vertex_ids`;
- face link: block `face_id -> neighbor_face_ids`;
- manifold edge link: one peer state per face edge;
- connected-component labels: one label per owning face/vertex carrier;
- domain labels: one label per face side.

Keep that key space explicit. A source-face provenance label is not a connected
component label.

## 7. Policies: attach capabilities to the right carrier

The pipe operator attaches compile-time capabilities:

```cpp
auto form = polygons
    | tf::tag(tree)
    | tf::tag(face_membership)
    | tf::tag(manifold_edge_link)
    | tf::tag(frame);
```

Algorithms detect those capabilities and select the reuse path.

### `tag` versus `zip`

- `tf::tag_normals(normals)` attaches a range-level normals carrier; the range
  exposes `.normals()`.
- `tf::zip_normals(normals)` additionally exposes `.normal()` on each element.
- `tf::zip(a, b)` is lazy lockstep composition, not parallel execution. Use
  `tf::parallel_for_each(tf::zip(a, b), ...)` for a parallel walk.

```cpp
auto points_with_normals = points | tf::zip_normals(normals);
auto normal = points_with_normals.front().normal();
```

### Tag lifetime

Spatial and topology tags borrow their owning structures. Keep the tree,
membership, links, half-edges, and source geometry alive while the tagged form
is used.

The same policy type is idempotent and first-wins. Retagging is not a refresh or
override operation:

```cpp
auto with_tree = polygons | tf::tag(tree0);
auto still_tree0 = with_tree | tf::tag(tree1);
```

Build a new form when a capability must be replaced.

## 8. Precompute, tag, and reuse

**Pre-tag every reusable dependency the planned pipeline consumes.** Build
dependencies in order so their builders also reuse earlier work.

```cpp
auto fm = tf::make_face_membership(polygons);
auto with_fm = polygons | tf::tag(fm);

auto mel = tf::make_manifold_edge_link(with_fm);
auto fl = tf::make_face_link(with_fm);

tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));

auto form = with_fm
    | tf::tag(mel)
    | tf::tag(fl)
    | tf::tag(tree);
```

Calling `make_manifold_edge_link(polygons)` after separately building `fm` would
miss the reuse path and can build membership again.

### Common capability sets

| Planned work | Carrier and reusable capabilities |
|---|---|
| Polygon spatial queries | polygon form + polygon AABB tree |
| Point search, ICP, Chamfer | points form + point AABB tree |
| Repeated topology analysis | polygons + FM, then links/labels built from tagged FM |
| Intersection/arrangement/boolean | polygon form + tree and MEL; retain/tag FM while building MEL and when other consumers use it |
| Repeated CSG operands | the same intersection capabilities, then one reusable CSG graph |
| Sharp edges | polygons + MEL + face normals |
| Curvature/smoothing | vertex link on points and the required point/face normals |
| Remeshing | triangle polygons + reusable half-edges |

Check the current function documentation for the exact tags it consumes. Do not
construct unrelated caches, but do not repeatedly invoke convenience builders
inside a pipeline.

One-shot convenience remains useful:

```cpp
auto boundary = tf::make_boundary_edges(polygons); // builds missing topology
```

Repeated work should carry the authority:

```cpp
auto boundary = tf::make_boundary_edges(with_fm);
auto non_manifold = tf::make_non_manifold_edges(with_fm);
auto closed = tf::is_closed(with_fm);
```

## 9. Lazy transformations and instances

A tagged transformation changes how algorithms observe geometry without
rewriting the local-coordinate buffers:

```cpp
auto base = polygons | tf::tag(tree);
auto instance0 = base | tf::tag(frame0);
auto instance1 = base | tf::tag(frame1);

auto distance = tf::distance(instance0, instance1);
```

The local-space tree is reused for both poses. This is the normal instancing and
moving-geometry path for spatial queries, intersections, arrangements, CSG,
remeshing metrics, registration, and transformed I/O.

An owning frame/transformation is copied or moved into the tag. Mutating an
external frame afterward does not update the tagged form. Update the frame held
by the tagged value, deliberately tag a view-backed frame, or create a new
tagged instance.

`tf::concatenated` is different: it applies tagged transformations and
materializes new owning world-space geometry.

## 10. Invalidation and updates

Capabilities are valid only for the geometry and identity space from which they
were built.

| Change | Invalidated state | Action |
|---|---|---|
| Point coordinates change | trees, normals, planes, metric-derived structures | rebuild immutable structures or update `mod_tree` |
| Faces/connectivity change | FM, links, half-edges, component/domain labels, polygon tree | rebuild/remap for the new carrier |
| Cleaning/reindexing/materialization | structures keyed by old IDs | use returned maps for attributes; rebuild structures on output |
| Only a lazy frame changes | local-space tree/topology remain valid | create/update the tagged instance |

Use `tf::tree` for immutable/rebuild workflows and `tf::mod_tree` when geometry
or topology is incrementally updated. A point tree and polygon tree are distinct
capabilities and belong at different carrier levels.

Never carry an old topology tag onto a new owning mesh just because coordinates
appear unchanged.

## 11. Results, labels, and index maps

High-level algorithms normally return owning buffers. Immediately named views
remain tied to those owners:

```cpp
auto curves = tf::make_intersection_curves(form0, form1);
auto paths = curves.paths();
auto curve_points = curves.points();
```

Optional policy arguments shape result tuples. Request provenance that the
producer already knows instead of reconstructing it later:

```cpp
auto [clean_mesh, face_map, point_map] =
    tf::cleaned(polygons, tf::return_index_map);

auto clean_face_values = tf::reindexed(face_values, face_map);
auto clean_point_values = tf::reindexed(point_values, point_map);
```

For ordinary `tf::index_map`:

- `f()[old_id]` is the new ID;
- `f().size()` is the removed sentinel;
- `kept_ids()[new_id]` is the retained old ID;
- merged duplicates can share one valid new ID.

Arrangement and CSG index-map structs have richer point/face/tag axes. Respect their
documented sentinels and boundary fields rather than assuming the basic map
shape.

Keep result meanings separate:

- `tag_labels`: source operand ID;
- `face_labels`: source face provenance;
- connected-component labels: connectivity grouping;
- domain labels: region on each side of a face.

## 12. Choose the right module-level operation

### Spatial

Spatial queries operate on tree-tagged forms:

```cpp
tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));
auto form = polygons | tf::tag(tree);

auto distance = tf::distance(form, query_point);
auto nearest = tf::neighbor_search(form, query_point, radius); // linear radius
auto hit = tf::ray_hit(ray, form, ray_config);
```

Prefer `gather_ids` for bulk ID collection. Manual `search` is for custom tree
traversal and requires thread-safe callback aggregation.

### Topology

Build dependency authorities once, then ask multiple questions:

```cpp
auto fm = tf::make_face_membership(polygons);
auto with_fm = polygons | tf::tag(fm);
auto mel = tf::make_manifold_edge_link(with_fm);
auto topology = with_fm | tf::tag(mel);

auto boundaries = tf::make_boundary_paths(topology);
auto components =
    tf::make_manifold_edge_connected_component_labels(topology);
auto [meshes, component_ids] =
    tf::split_into_components(polygons, components);
```

Do not pass boolean/arrangement `face_labels` to `split_into_components` and
describe the result as connectivity components; those labels group by source
face.

Use `tf::triangulated(polygons)` for an owning triangle mesh; it returns the
corners with the point table they index, because a resolved face mints
identities the input's own table has no row for. It takes an indexed mesh, a
single `tf::polygon`, or a SOUP — a soup is `tf::cleaned` to shared-vertex
identity first, so the triangulation machinery only ever sees indexed meshes
and a shared edge is one identity there as anywhere else.

Its leading template parameter answers two questions: the width the faces are
written in, when an input's index type is wider than the caller needs; and the
name of the output's index type, when the input is a soup and has none (there
it defaults to `int`, elsewhere to the input's own). `tf::return_refused` takes
indexed meshes only — a soup's faces do not survive the clean, so it has no
face identity to refuse.

Connectivity without a mesh is the arrangement mesh tier's own product
(`tf::arrangement::make_mesh_triangulation`), not a public entry.

### Geometry

Geometry factories and analyses return owning buffers:

```cpp
auto normals = tf::compute_normals(polygons);
auto sharp = tf::make_sharp_edges(
    polygons | tf::tag(mel) | tf::tag_normals(normals.unit_vectors()),
    tf::deg(30.f));
```

Registration and Chamfer search require a target point tree:

```cpp
tf::aabb_tree<int, float, 3> target_tree(target_points,
                                         tf::config_tree(4, 4));
auto target = target_points | tf::tag(target_tree);
auto delta = tf::fit_icp_alignment(source_points | tf::tag(initial), target,
                                   config);
auto total = tf::transformed(initial, delta);
auto error = tf::chamfer_error(source_points | tf::tag(total), target);
```

Chamfer error is directional; compute both directions for a symmetric metric.
Smoothing consumes a vertex link tagged on the point carrier.

### Remesh

High-level remesh functions require triangle polygons and return a new owning
mesh plus resulting half-edges:

```cpp
auto triangles = tf::triangulated(polygons);
tf::half_edges<int> he(triangles.polygons());

auto [decimated, decimated_he] =
    tf::decimated(triangles.polygons() | tf::tag(he), 0.1f);
```

Choose by stopping rule:

- `decimated`: target face fraction;
- `simplified`: geometric error budget;
- `isotropic_remeshed`: target edge length;
- `collapsed_short_edges`: length threshold.

Set `config.parallel = false` when an outer application already parallelizes
many meshes. Low-level overloads operating on half-edges and point buffers can
mutate those objects in place; do not confuse them with functional high-level
returning overloads.

### Intersect, Arrangement, and CSG

Choose the cheapest materialization that answers the question:

| Need | Operation |
|---|---|
| Intersection polylines only | `make_intersection_curves` |
| Complete cut surface and provenance | mesh/polygon arrangements |
| One boolean result | `make_boolean` |
| Many expressions/domains/shells/seams over fixed operands | one `csg_graph`, many extractors |

For repeated intersection/arrangement work, tag every operand's reusable
structures:

```cpp
auto fm = tf::make_face_membership(polygons);
auto with_fm = polygons | tf::tag(fm);
auto mel = tf::make_manifold_edge_link(with_fm);
tf::aabb_tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));
auto form = with_fm | tf::tag(mel) | tf::tag(tree);
```

Boolean/CSG inclusion expects locally consistently oriented PWN inputs. Select
`intersect_mode`, tolerance, and arrangement triangulation for the input
semantics; do not treat configuration as a performance-only knob.

Build one CSG graph for repeated extraction:

```cpp
std::array forms{form0, form1, form2};
auto forms_range = tf::make_range(forms);
auto graph = tf::make_csg_graph(forms_range);

auto mesh = tf::make_csg_mesh(
    graph, tf::csg::op(0) - tf::csg::merge(1, 2));
auto full_arrangement = tf::make_csg_mesh(graph);
auto seams = tf::make_intersection_curves(graph);
auto [cells, ids] = tf::make_csg_domains(graph);
```

The source buffers, forms container/range, and externally tagged structures
must outlive the graph. Direct intersection-curve calls perform a fresh
intersection; the graph overload reuses the stored arrangement.

### Clean and Reindex

Use cleaning to create or repair identity, and request maps when aligned data
must follow:

```cpp
auto [cleaned, face_map, point_map] =
    tf::cleaned(polygons, tolerance, tf::return_index_map);
auto filtered = tf::reindexed_by_mask(cleaned.polygons(), face_mask);
```

Soup cleaning creates indexed geometry but has no meaningful source index map.
Curve cleaning returns only a point map because reconnecting paths changes edge
identity.

`concatenated` materializes one owning carrier and offsets indices. Mixed face
arity produces dynamic polygons. `split_into_components` groups by the labels
you supply; it does not infer connectivity by itself.

### I/O

Readers return owning buffers:

```cpp
auto stl = tf::read_stl("mesh.stl");       // indexed float triangles
auto obj = tf::read_obj("mesh.obj");       // dynamic polygons, positions
auto complete = tf::read_obj("mesh.obj", tf::complete);
```

Complete OBJ mode aligns position/normal/texture tuples and duplicates vertices
at attribute seams. Writers accept tagged transformations and materialize them
in the file. STL output requires 3D triangles. Check the returned success/empty
state and use the buffer-based I/O overloads when files are not the ownership
boundary.

## 13. Caller checklist

Before calling a nontrivial pipeline, verify:

1. Is semantic geometry represented by Trueform primitives/views?
2. Which object owns every buffer viewed by the input?
3. Is the carrier a point, segment, polygon, curve, face, vertex, or domain?
4. Is static arity preserved?
5. Does topology share intentional point IDs?
6. Which reusable structures does the complete pipeline consume?
7. Were dependencies built in order and tagged on the correct carrier?
8. Will every tagged structure outlive the call or graph?
9. Has geometry/connectivity changed since the structures were built?
10. Is a transformation lazy, or is world-space materialization required?
11. Which optional output policy preserves provenance or attributes?
12. Are labels interpreted by their real carrier rather than their name?
13. Is the chosen operation the cheapest materialization that answers the
    question?

## Module references

- Core model: `docs/content/cpp/2.modules/01.core.md`
- Spatial: `docs/content/cpp/2.modules/02.spatial.md`
- Topology: `docs/content/cpp/2.modules/03.topology.md`
- Geometry: `docs/content/cpp/2.modules/04.geometry.md`
- Remesh: `docs/content/cpp/2.modules/05.remesh.md`
- Intersect: `docs/content/cpp/2.modules/06.intersect.md`
- Arrangement: `docs/content/cpp/2.modules/07.arrangement.md`
- Iso: `docs/content/cpp/2.modules/08.iso.md`
- CSG: `docs/content/cpp/2.modules/09.csg.md`
- Clean: `docs/content/cpp/2.modules/10.clean.md`
- Reindex: `docs/content/cpp/2.modules/11.reidx.md`
- I/O: `docs/content/cpp/2.modules/12.io.md`
