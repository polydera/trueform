---
title: Examples
description: Core functionality, comparisons, and VTK integration.
---

The examples are grouped into three categories. Build them with CMake and explore practical programs and benchmarks.

```bash [Terminal]
mkdir build
cd build
cmake ..
make examples -j8
```


## Core Functionality

- **Queries on Primitives**: geometric queries between low-level primitives.  
  `examples/queries_on_primitives.cpp`
- **Finding Coincident Points with Tolerance**: gather near-duplicates with `neighbor_search`.  
  `examples/find_coincident_points_with_tolerance.cpp`
- **Find Intersecting Primitives in a Mesh**: self-collision via `gather_ids` and forms.  
  `examples/find_intersecting_primitives.cpp`

## Comparisons

- **nanoflann: k-NN Queries**: benchmarks vs `nanoflann`.  
  `examples/nanoflann/knn.cpp`
- **CGAL: Intersecting Primitives**: pairs via `gather_ids` vs CGAL.  
  `examples/cgal/intersecting_primitives.cpp`
- **CGAL: Intersection Curve**: intersection curves between transforming meshes.  
  `examples/cgal/intersection_curve.cpp`
- **CGAL: Planar Arrangements**: faces from unordered 2D segments.  
  `examples/cgal/planar_arrangments.cpp`

## VTK Integration

- **Actor Picking and Collision Detection**: interactive collision with forms.  
  `examples/vtk/collision.cpp`
- **Automatic Positioning**: closest-points placement using `neighbor_search`.  
  `examples/vtk/positioning.cpp`
- **Scalar-Field Intersection Curve**: slicing meshes by scalar fields.  
  `examples/vtk/scalar_field_intersections.cpp`
- **Mesh-Mesh Intersection Curve**: dynamic intersection curves between forms.  
  `examples/vtk/forms_intersections.cpp`
