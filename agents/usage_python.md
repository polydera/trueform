# Python Usage Patterns

> **Lookup only.** This file describes public Python usage, not binding or core
> implementation strategy. Verify names against the current package and read
> `python_layer.md` before changing bindings.

```python
import trueform as tf
import numpy as np
```

---

## 1. Primitives

```python
# Point
p = tf.Point([1.0, 2.0, 3.0])
p.coords  # [1. 2. 3.]
p.x, p.y, p.z  # 1.0, 2.0, 3.0
p.dims   # 3
p.dtype  # float32

# Batch of points
pts = tf.Point(np.random.rand(100, 3).astype(np.float32))
pts.count  # 100
pts.is_batch  # True

# Vector, Unit Vector
v = tf.Vector([1.0, 0.0, 0.0])

# Segment
seg = tf.Segment([[0, 0, 0], [1, 1, 1]])
seg.start, seg.end, seg.length, seg.midpoint

# Batch of segments
segs = tf.Segment(np.random.rand(50, 2, 3).astype(np.float32))

# Ray
ray = tf.Ray(origin=[0, 0, 0], direction=[1, 0, 0])
ray = tf.Ray.from_points(start=[0, 0, 0], through_point=[1, 1, 1])

# Line
line = tf.Line(origin=[0, 0, 0], direction=[1, 0, 0])
line = tf.Line.from_points([0, 0], [1, 1])

# Triangle
tri = tf.Triangle(a=[0, 0, 0], b=[1, 0, 0], c=[0, 1, 0])
tri = tf.Triangle([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
tf.area(tri)    # 0.5
tf.normals(tri) # [0. 0. 1.]

# Polygon
quad = tf.Polygon([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
tf.area(quad)   # 1.0

# Plane
plane = tf.Plane(normal=[0, 0, 1], offset=-5.0)
plane = tf.Plane(normal=[0, 0, 1], origin=[0, 0, 5])
plane = tf.Plane.from_points([0,0,0], [1,0,0], [0,1,0])

# AABB
aabb = tf.AABB(min=[0, 0, 0], max=[1, 1, 1])
aabb = tf.AABB.from_center_size(center=[5, 5], size=[2, 2])
aabb = tf.AABB.from_points([[0, 0], [1, 0], [1, 1]])
aabb.center, aabb.size, aabb.volume

# Batch of any primitive
boxes = tf.AABB(min=np.zeros((50, 3), dtype=np.float32),
                max=np.ones((50, 3), dtype=np.float32))
```

---

## 2. Forms (Mesh, PointCloud, EdgeMesh)

### Mesh

```python
# Triangle mesh
faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
points = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]], dtype=np.float32)
mesh = tf.Mesh(faces, points)

mesh.is_dynamic  # False
mesh.ngon        # 3

# Dynamic mesh (variable-size faces)
offsets = np.array([0, 4, 7], dtype=np.int32)
data = np.array([0,1,2,3, 4,5,6], dtype=np.int32)
faces = tf.OffsetBlockedArray(offsets, data)
mesh = tf.Mesh(faces, points)

mesh.is_dynamic  # True
mesh.ngon        # None

# From file
mesh = tf.Mesh(*tf.read_stl("model.stl"))
```

### PointCloud

```python
cloud = tf.PointCloud(np.random.rand(100, 3).astype(np.float32))
cloud.size   # 100
cloud.dims   # 3
cloud.dtype  # float32
```

### EdgeMesh

```python
edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
points = np.array([[0,0], [1,0], [1,1]], dtype=np.float32)
edge_mesh = tf.EdgeMesh(edges, points)
```

### Topology (Lazy, Cached)

Topology structures are computed on first access and cached internally. No manual rebuild needed — if you change faces or points, caches are invalidated.

```python
mesh.face_membership     # computed on first access, cached
mesh.manifold_edge_link  # cached
mesh.face_link           # cached
mesh.vertex_link         # cached

# Tree is also lazy — built on first spatial query or explicit call
mesh.build_tree()
```

### Transformations

Transformations are applied at query time — the underlying data stays in local coordinates. This enables shared views with different poses without copying geometry or rebuilding trees.

```python
# 4x4 homogeneous transformation
T = np.eye(4, dtype=np.float32)
T[:3, 3] = [5, 0, 0]

# Set at construction or later
mesh = tf.Mesh(faces, points, transformation=T)
mesh.transformation = T

# Shared views: same data + tree, different pose per instance
mesh.build_tree()
view_a = mesh.shared_view()
view_a.transformation = transform_A
view_b = mesh.shared_view()
view_b.transformation = transform_B
# Both share the same tree — only the transform differs
d = tf.distance(view_a, view_b)
```

### Transforming Primitives

```python
T = np.eye(4, dtype=np.float32)
T[:3, 3] = [10, 0, 0]
transformed_pt = tf.transformed(tf.Point([1, 2, 3]), T)
```

---

## 3. Spatial Queries

### Distance

```python
# Primitive × Primitive
d = tf.distance(tf.Point([0,0,0]), tf.Segment([[1,0,0], [1,1,0]]))

# Form × Primitive
d = tf.distance(mesh, tf.Point([0, 0, 0]))

# Form × Form
d = tf.distance(mesh_a, mesh_b)

# Batch → array
pts = tf.Point(np.random.rand(1000, 3).astype(np.float32))
distances = tf.distance(pts, mesh)  # shape (1000,)
```

### Closest Point

```python
dist2, pt = tf.closest_metric_point(polygon, point)
dist2, pt_a, pt_b = tf.closest_metric_point_pair(seg_a, seg_b)
```

### Neighbor Search

```python
# Single → (element_id, dist², closest_point)
idx, dist2, pt = tf.neighbor_search(mesh, point)

# With radius
result = tf.neighbor_search(mesh, point, radius=1.0)
if result is not None:
    idx, dist2, pt = result

# Batch → arrays
ids, dist2s, pts = tf.neighbor_search(mesh, batch_points)

# Form × Form
(id0, id1), (dist2, pt0, pt1) = tf.neighbor_search(mesh_a, mesh_b)

# kNN
neighbors = tf.neighbor_search(mesh, point, k=10)
for idx, dist2, pt in neighbors: pass

# Batch kNN
ids, dist2s, pts, counts = tf.neighbor_search(mesh, batch_pts, k=10)
# ids.shape = (N, K), counts[i] = valid count for query i
```

### Ray Casting

```python
config = (0.0, 100.0)  # (min_t, max_t)

# Single
result = tf.ray_cast(ray, mesh, config)
if result is not None:
    face_id, t = result

# Batch
ids, ts = tf.ray_cast(batch_rays, mesh, config)  # -1/NaN for misses

# Per-ray config
ids, ts = tf.ray_cast(rays, mesh, config=(min_ts, max_ts))
```

### Intersection Tests

```python
hit = tf.intersects(mesh, polygon)                   # bool
hit = tf.intersects(mesh_a, mesh_b)                  # bool
hits = tf.intersects(mesh, batch_segs)               # array of bool
```

### Gathering IDs

```python
ids = tf.gather_intersecting_ids(mesh, polygon)       # 1D face IDs
ids = tf.gather_ids_within_distance(mesh, point, distance=0.5)
pairs = tf.gather_intersecting_ids(mesh_a, mesh_b)    # (M, 2) ID pairs
```

---

## 4. Boolean Operations

```python
# Union / Intersection / Difference
(faces, points), labels, face_labels = tf.boolean_union(mesh0, mesh1)
(faces, points), labels, face_labels = tf.boolean_intersection(mesh0, mesh1)
(faces, points), labels, face_labels = tf.boolean_difference(mesh0, mesh1)

# With curves
(faces, points), labels, face_labels, (paths, curve_pts) = tf.boolean_union(
    mesh0, mesh1, return_curves=True)

# Mesh arrangements
(faces, points), tag_labels, face_labels = tf.mesh_arrangements([mesh0, mesh1])
(faces, points), tag_labels, face_labels, (paths, curve_pts) = tf.mesh_arrangements(
    [mesh0, mesh1], return_curves=True)

# Embedded intersection curves
(faces, points), face_labels = tf.embedded_intersection_curves(mesh0, mesh1)
(faces, points), face_labels = tf.embedded_self_intersection_curves(mesh)

# Isobands
(faces, points), labels, face_labels = tf.isobands(mesh, scalars, [-1.0, 0.0, 1.0])
(faces, points), labels, face_labels, (paths, curve_pts) = tf.isobands(
    mesh, scalars, cut_values, return_curves=True, selected_bands=[1, 3])
```

### CSG graph: build once, query many (N-ary)

```python
graph = tf.CsgGraph([a, b, c], sheets=[2], triangulation="refined_cdt")

faces, points = graph.mesh(tf.op(0) - tf.op(1))       # any expression: | & - ~
full = graph.mesh()                                    # full arrangement mesh
(f, p), tags, fl = graph.mesh(tf.op(0) | 1, return_source_ids=True)
(f, p), imap = graph.mesh(tf.op(0) - 1, return_index_map=True)

cells, ids = graph.domains()                           # every kept domain
paths, curve_points = graph.intersection_curves()
graph.forms; graph.sheets; graph.created_points        # remembered state
```


---

## 5. Topology

```python
# Connected components
num_components, labels = tf.label_connected_components(mesh.face_link)
components, comp_ids = tf.split_into_components(mesh, labels)

# Boundary
paths, boundary_points = tf.boundary_curves(mesh)
bad_edges = tf.non_manifold_edges(mesh)

# Vertex neighborhoods
k2_ring = tf.k_rings(mesh.vertex_link, k=2)
neighs = tf.neighborhoods(mesh.vertex_link, mesh.points, radius=0.5)
```

---

## 6. Geometry

```python
# Normals
face_normals = tf.normals(mesh)
point_normals = tf.point_normals(mesh)

# Curvatures
k0, k1, d0, d1 = tf.principal_curvatures(mesh, directions=True)

# Registration
transform = tf.fit_icp_alignment(source, target)
transform = tf.fit_rigid_alignment(source, target)
error = tf.chamfer_error(source, target)
```

---

## 7. Mesh Processing

```python
# Clean
clean_faces, clean_points = tf.cleaned((faces, points))

# Triangulate
tri_faces, tri_points = tf.triangulated((quad_faces, points))

# Orient
new_faces = tf.ensure_positive_orientation(mesh)

# Decimate + Remesh
dec_faces, dec_points = tf.decimated(mesh, 0.5)
dec_mesh = tf.Mesh(dec_faces, dec_points)
mel = tf.mean_edge_length(dec_mesh)
rem_faces, rem_points = tf.isotropic_remeshed(dec_mesh, mel)
```

---

## 8. I/O

```python
faces, points = tf.read_stl("model.stl")
faces, points = tf.read_obj("model.obj")
tf.write_stl((faces, points), "output.stl")
tf.write_obj((faces, points), "output.obj")
```

---

## 9. OffsetBlockedArray

```python
offsets = np.array([0, 3, 7, 10], dtype=np.int32)
data = np.array([0,1,2, 3,4,5,6, 7,8,9], dtype=np.int32)
blocks = tf.OffsetBlockedArray(offsets, data)

len(blocks)     # 3
blocks[0]       # array([0, 1, 2])
blocks[1]       # array([3, 4, 5, 6])

for block in blocks:
    print(block)

# From uniform arrays
quads = np.array([[0,1,2,3], [4,5,6,7]], dtype=np.int32)
faces = tf.as_offset_blocked(quads)
```

---

## 10. Broadcasting Rules

```python
# single × single → scalar
d = tf.distance(tf.Point([0,0,0]), tf.Point([1,0,0]))

# batch × single → array
distances = tf.distance(batch_points, plane)  # (N,)

# batch × batch → array (must have same count)
distances = tf.distance(pts_a, pts_b)         # (N,)

# Batch properties
segs = tf.Segment(np.random.rand(50, 2, 3).astype(np.float32))
segs.start.shape     # (50, 3)
segs.length.shape    # (50,)
segs.midpoint.shape  # (50, 3)
```

---

## 11. Data Types

Python trueform supports these numpy dtype combinations:

| Index Type | Real Type | Dimensions |
|-----------|-----------|------------|
| `np.int32` | `np.float32` | 2D, 3D |
| `np.int32` | `np.float64` | 2D, 3D |
| `np.int64` | `np.float32` | 2D, 3D |
| `np.int64` | `np.float64` | 2D, 3D |

The correct C++ overload is selected automatically based on input dtypes.
