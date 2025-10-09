# Examples

The examples are grouped into three main categories:

* [Core Functionality](#core-functionality): Self-contained examples that demonstrate the primary features of the trueform library, from primitive queries to the compositional policy system.
* [Comparisons](#comparisons): Performance and usage comparisons against other well-established libraries like `VTK`, `CGAL` and `nanoflann`, showcasing the efficiency of trueform's data structures and algorithms.
* [VTK Integration](#vtk-integrationvtk): Practical guides on how to integrate `trueform` into a larger framework like `VTK`, demonstrating how to create `trueform` views directly over `VTK`'s data structures.

To build examples, run:

```bash
mkdir build
cd build
cmake ..
make examples -j8
```

## Core Functionality

This directory contains a set of standalone examples demonstrating various features and use cases of the trueform library. Each example is designed to be a practical, focused demonstration of a specific workflow.

### [Queries on Primitives](./queries_on_primitives.cpp)

This example demonstrates how to perform fundamental geometric queries between individual, low-level primitives. It serves as a basic introduction to the core query system.

#### Features Showcased:

* Creating basic owning primitives like `tf::polygon` and `tf::segment`.
* Performing boolean intersection tests with `tf::intersects`
* Finding the closest points between two primitives with `tf::closest_metric_point_pair`
* Calculating the minimum distance with `tf::distance`
* Classifying points against polygons, planes and lines using `tf::classify` to get `tf::containment` and `tf::sidedness` results.
* Casting rays against primitives with `tf::ray_cast` and `tf::ray_hit.`

### [Finding Coincident Points with Tolerance](./find_coincident_points_with_tolerance.cpp)

This example demonstrates a common and practical geometry processing workflow: cleaning up a point cloud by identifying nearly-coincident points. This is a crucial pre-processing step for many algorithms that require "welded" geometry, such as mesh simplification or manifold topology construction.

#### Features Showcased:

* Generating a random `tf::points` collection.
* Building a `tf::tree` acceleration structure over a point cloud.
* Using the high-level `tf::gather_self_ids` query to collect all pairs of points within a given tolerance.
* Using the more general tf::search_self query to perform the same task, demonstrating how to use the flexible broad-phase and narrow-phase lambdas.
* Using `tf::local_vector` for thread-safe parallel collection.

### [Find Intersecting Primitives in a Mesh](./find_intersecting_primitives.cpp)
This example demonstrates how to find all intersecting triangles between a mesh and a dynamically transformed version of itself. This is a core task for collision detection with articulated bodies (e.g., a robot arm colliding with itself) or general self-intersection tests.

#### Features Showcased:

* Creating a `tf::polygons` view from raw vertex and index data.
* Building a `tf::tree` over a polygon mesh.
* Applying a dynamic `tf::frame` to a `tf::form` to perform a query on transformed geometry without modifying the original data.
* Using the high-level `tf::gather_ids` to quickly collect all intersecting pairs.
* Using the general-purpose tf::search to perform the same query, showing how to use `tf::intersects` as a predicate for both the broad-phase and narrow-phase checks.
* Using `tf::local_vector` for thread-safe parallel collection.

## Comparisons

These examples contain a direct comparison against well-established libraries.

### [nanoflann: k-NN Queries](./nanoflann/knn.cpp)

This example provides a direct performance benchmark for k-Nearest Neighbor (k-NN) queries, comparing trueform's tf::tree against nanoflann. It demonstrates how to create a nanoflann-compatible adaptor for trueform data structures.

#### Features Showcased:

* Building a `tf::tree` and a `nanoflann::KDTree` from the same data source.
* Benchmarking k-NN query performance for both libraries over a range of k values.
* Demonstrating the performance advantage of trueform's spatial query system.

### [CGAL: Intersecting Primitives](./cgal/intersecting_primitives.cpp)

This example provides direct performance benchmark for collecting intersecting primitives, comparing trueform's gather_ids against CGAL.

**NOTE:** CGALS call `all_intersecting_primitives` produces less information and terminates early, as it only returns primitives on the first mesh that intersect any primitive on the second mesh, while trueform's `tf::gather_ids` returns all intersecting primitive pairs. Hence it does more work in less time.

#### Features Showcased:

* Building a `tf::tree`
* Transformations of a `tf::tree`
* Using `tf::gather_ids`


### [CGAL: Intersection Curve](./cgal/intersection_curve.cpp)

This example provides a direct comparison with `CGAL` for computing the intersection curve between two transforming meshes.

#### Features Showcased

* Using `tf::polygons_intersections` to compute intersections between two `tf::form`s.
* Using `tf::make_intersection_edges` to extract intersection edges from `tf::polygons_intersections`
* Using `tf::face_membership` and how to `tf::tag` it to a `tf::form` 
* Using `tf::manifold_edge_link` and how to `tf::tag` it to a `tf::form` 
* Using `points.as<double>()` to get a view into points where each point is cast to `tf::point<double,3>`

### [CGAL:  Planar Arrangments](./cgal/planar_arrangments.cpp)

This example provides a direct comparison with `CGAL` for computing [planar arrangments](https://doc.cgal.org/latest/Arrangement_on_surface_2/index.html) of on an unordered collection of edges.

#### Features Showcased

* Using `tf::planar_arrangments` to extract oriented faces and hole relations.
* Using `tf::make_edges` and `tf::make_points` to impart semantic meaning onto ranges

### [VTK: Feature Edges](./vtk/feature_edges.cpp)

This example provides direct performance benchmark for collecting boundary and non_manifold edges.

#### Features Showcased

* Using `tf::make_boundary_edges`
* Using `tf::make_non_manifold_edges`

### [VTK: Clean Mesh](./vtk/compare_clean.cpp.cpp)

This example provides direct performance benchmark for cleaning a mesh with a specified tolerance (our implementation additionally removes all uncontained points).

#### Features Showcased

* Using `tf::cleaned`

### [VTK: Intersection Curve](./vtk/vtk_intersection_curve.cpp)

This example provides a direct comparison with `VTK` for computing the intersection curve between two transforming meshes.

#### Features Showcased

* Using `tf::polygons_intersections` to compute intersections between two `tf::form`s.
* Using `tf::make_intersection_edges` to extract intersection edges from `tf::polygons_intersections`
* Using `tf::face_membership` and how to `tf::tag` it to a `tf::form` 
* Using `tf::manifold_edge_link` and how to `tf::tag` it to a `tf::form` 
* Using `points.as<double>()` to get a view into points where each point is cast to `tf::point<double,3>`

## [VTK Integration](./vtk/)

This set of examples demonstrates how trueform can be integrated into a larger framework like VTK. It shows how to create trueform views directly over VTK's data structures, perform complex queries, and handle different data layouts (e.g., the transition from VTK 7 to 8+).

### [Actor Picking and Collision Detection](./vtk/collision.cpp)

This example demonstrates how to integrate trueform into an interactive VTK application. It loads one or more user-provided meshes and arranges them in a 5x5 grid (with repetition and random rotations). The user can pick and drag an actor with the mouse, and trueform performs real-time collision detection between the moving actor and the static grid, highlighting any colliding objects.

#### Features Showcased:

* Creating `tf::polygons` views directly over VTK data buffers.
* Building a `tf::tree` over a collection of objects.
* Using `tf::intersects`` with a dynamic `tf::form` to detect collisions 
* Using `tf::ray_cast` for mouse picking.

### [Automatic Positioning](./vtk/positioning.cpp)
This example showcases a common task in assembly or scene layout: precisely positioning two objects. It loads two copies of a mesh randomly rotated, allowing the user to move them freely. It then uses `trueform`'s `tf::neighbor_search` to find the closest points between the two meshes and calculates the exact transformation needed to bring them into contact along the shortest path.

#### Features Showcased:

* Using `tf::neighbor_search` between two `tf::form` objects to find the closest points on complex meshes.
* Using the result of a query to compute a precise placement `tf::frame`.
* Demonstrating a practical application for CAD, robotics, or virtual assembly.

### [Scalar-Field Intersection Curve](./vtk/scalar_field_intersections.cpp)

This example showcases how to compute intersection curves on a mesh, based on a scalar field on its points. It loads a mesh and picks a random plane. When you scroll the mouse wheel, it adjust the plane in the normal direction in a continous way. Pressing `n` randomizes the plane.

#### Features Showcased

* Using `tf::scalar_field_intersections`
* Using `tf::make_intersection_edges` 
* Demonstrates a practical application for CAD

### [Mesh-Mesh Intersection Curve](./vtk/forms_intersections.cpp)

This example showcases how to compute intersection curves between two meshes (as `tf::form`). It loads two meshes (or one and duplicates it). You interact with the scene by grabbing one of the meshes and move it. When the `forms` are intersecting, the intersection curve will be shown. Pressing `n` randomizes the rotations of the meshes.

#### Features Showcased

* Using `tf::polygons_intersections` to compute intersections between two `tf::form`s.
* Using `tf::make_intersection_edges` to extract intersection edges from `tf::polygons_intersections`
* Using `tf::face_membership` and how to `tf::tag` it to a `tf::form` 
* Using `tf::manifold_edge_link` and how to `tf::tag` it to a `tf::form` 
* Using `points.as<double>()` to get a view into points where each point is cast to `tf::point<double,3>`
* Demonstrates a practical application for CAD
