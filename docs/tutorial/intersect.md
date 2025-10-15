# Intersect

The `intersect` module computes geometric intersections between meshes and scalar fields, extracting intersection curves and points with robust handling of all topological cases.

## Intersection Curves

The simplest way to work with intersections is through curves - connected paths of intersection points.

### Between Two Meshes

Extract intersection curves where two meshes intersect:

```cpp
// Load meshes
tf::polygons_buffer<int, float, 3, 3> mesh1, mesh2;
// ... fill with data ...

// Prepare topology
tf::face_membership<int> fm1, fm2;
fm1.build(mesh1.polygons());
fm2.build(mesh2.polygons());

tf::manifold_edge_link<int, 3> mel1, mel2;
mel1.build(mesh1.faces(), fm1);
mel2.build(mesh2.faces(), fm2);

// Build spatial forms
tf::tree<int, float, 3> tree1(mesh1.polygons(), tf::config_tree(4, 4));
tf::tree<int, float, 3> tree2(mesh2.polygons(), tf::config_tree(4, 4));

auto form1 = tf::make_form(tree1, mesh1.polygons() | tf::tag(mel1) | tf::tag(fm1));
auto form2 = tf::make_form(tree2, mesh2.polygons() | tf::tag(mel2) | tf::tag(fm2));

// Extract intersection curves
auto curves = tf::make_intersection_curves(form1, form2);
```

Use the curves:

```cpp
// Access as tf::curves
for (auto curve : curves.curves()) {
}

auto& paths = curves.paths();
auto& points = curves.points();

```

### Self-Intersection Curves

Find where a mesh intersects itself:

```cpp
// Single mesh with topology
tf::face_membership<int> fm;
fm.build(mesh.polygons());

tf::manifold_edge_link<int, 3> mel;
mel.build(mesh.faces(), fm);

tf::tree<int, float, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
auto form = tf::make_form(tree, mesh.polygons() | tf::tag(mel) | tf::tag(fm));

// Extract self-intersection curves
auto self_curves = tf::make_self_intersection_curves(form);

// Use identically to regular intersection curves
for (auto curve : self_curves.curves()) {
    // Process self-intersection curve
}
```

## Isosurface Curves (Isocontours)

Extract curves where a scalar field crosses threshold values:

### Single Isocontour

```cpp
// Define scalar field over mesh vertices
std::vector<float> scalar_field{mesh.points().size()};
tf::parallel_transform(mesh.points(),
                      distance_field,
                      tf::distance_f(tf::make_plane(triangles.front()))

// Extract isocontour at threshold
float threshold = 0.5f;
auto contour = tf::make_isocontours(mesh.polygons(), scalar_field, threshold);

// Use as curves
for (auto curve : contour.curves()) {
}
```

### Multiple Isocontours

Efficiently compute multiple isosurface levels:

```cpp
// Define threshold levels
std::vector<float> levels = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

// Compute all levels at once
auto contours = tf::make_isocontours(mesh.polygons(), scalar_field, levels);

// Each curve belongs to one threshold level
for (auto curve : contours.curves()) {
    // Connected isoline at some threshold
}
```

## Low-Level Intersection Access

For advanced use cases, direct access to intersection data:

### Between Two Meshes

```cpp
tf::intersections_between_polygons<int, double, 3> ibp;
ibp.build(form1, form2);

// Access intersection points
auto points = ibp.intersection_points();
for (auto point : points) {
    // Each point is a tf::point<double, 3>
}

// Access structured intersections
for (auto group : ibp.intersections()) {
    // Each group contains intersections for one face
    for (auto intersection : group) {
        // Intersection fields:
        intersection.tag;           // 0 or 1 (which object)
        intersection.object;        // Face ID in first mesh
        intersection.object_other;  // Face ID in second mesh
        intersection.id;            // Point ID in intersection_points()

        // Topological location on first mesh
        intersection.target.label;  // tf::topo_type: vertex/edge/face
        intersection.target.id;     // Local vertex/edge index

        // Topological location on second mesh
        intersection.target_other.label;
        intersection.target_other.id;
    }
}

// Separate by mesh
auto intersections0 = ibp.intersections0();  // First mesh faces
auto intersections1 = ibp.intersections1();  // Second mesh faces
```

### Self-Intersections

```cpp
tf::intersections_within_polygons<int, double, 3> iwp;
iwp.build(form);

auto points = iwp.intersection_points();

for (auto group : iwp.intersections()) {
    for (auto intersection : group) {
        intersection.object;        // Face ID
        intersection.object_other;  // Other intersecting face ID
        intersection.id;            // Point ID
        intersection.target.label;  // Topological type
        intersection.target.id;     // Local index
        intersection.target_other.label;
        intersection.target_other.id;
    }
}
```

### Segment Intersections

For 2D line segment collections:

```cpp
// Prepare segments with edge membership
auto segments = tf::make_segments(edges, points) | tf::tag(tf::edge_membership);

tf::tree<int, float, 2> tree;
tree.build(segments, tf::config_tree(4, 4));

// Compute intersections
tf::intersections_within_segments<int, float, 2> iws;
iws.build(segments, tree);

auto points = iws.intersection_points();
auto groups = iws.intersections();

// Access structured intersections
for (auto group : ibp.intersections()) {
    // Each group contains intersections for one edge
    for (auto intersection : group) {
        intersection.object;        // Edge ID
        intersection.object_other;  // Other intersecting Edge ID
        intersection.id;            // Point ID
        intersection.target.label;  // Topological type
        intersection.target.id;     // Local index
        intersection.target_other.label;
        intersection.target_other.id;
}
```

### Scalar Field Intersections

Low-level access to isosurface intersections:

```cpp
tf::scalar_field_intersections<int, float, 3> sfi;
sfi.build(mesh.polygons(), scalar_field, threshold);

// Access points where field crosses threshold
auto points = sfi.intersection_points();

// Access structured intersections
for (auto group : sfi.intersections()) {
    // Intersections grouped by face
    for (auto intersection : group) {
        intersection.object;       // Face ID
        intersection.id;           // Point ID
        intersection.target.label; // vertex/edge where crossing occurs
        intersection.target.id;    // Local index
    }
}

// Multiple thresholds
sfi.build_many(mesh.polygons(), scalar_field, thresholds);
```
