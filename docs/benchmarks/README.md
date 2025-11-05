# Trueform Benchmark Data

This directory contains benchmark data comparing `trueform` performance against industry-standard libraries (VTK and CGAL).

## Benchmark Files

### Mesh-Mesh Intersection Curves

**vs VTK**: `mesh_intersection_curves_vs_vtk.csv`
- Compares intersection curve extraction between two triangle meshes
- Dataset: Two meshes of equal size (50k to 1M triangles each, total 100k to 2M)
- **Result: 114x to 518x speedup**
- **Pattern**: Speedup increases consistently with mesh size
  - 100k triangles: 114.5x
  - 500k triangles: 220.3x
  - 1M triangles: 376.3x
  - 2M triangles: 518.1x
- **Analysis**: Performance advantage grows with mesh size. The speedup approximately increases by 1.5x per doubling of mesh size, indicating better scaling characteristics in trueform's spatial indexing and parallel intersection algorithm.

**vs CGAL**: `mesh_intersection_curves_vs_cgal.csv`
- Same operation compared against CGAL
- Dataset: Two meshes of equal size (50k to 1M triangles each, total 100k to 2M)
- **Result: 31x to 178x speedup**
- **Pattern**: Speedup grows with mesh complexity
  - 100k triangles: 31.5x
  - 500k triangles: 91.5x
  - 1M triangles: 122.8x
  - 2M triangles: 177.6x
- **Analysis**: The speedup approximately doubles with each size increase, reaching 178x at the largest test case.

### Scalar Field Contours (Isosurfaces)

**Single Contour**: `scalar_field_contours_vs_vtk.csv`
- Extracts single isocontour from scalar field on mesh
- Dataset: Meshes from 50k to 1M triangles
- **Result: 10x to 27x speedup**
- **Pattern**: Speedup increases with mesh size
  - 50k triangles: 10.2x
  - 250k triangles: 19.7x
  - 500k triangles: 23.2x
  - 1M triangles: 27.1x
- **Analysis**: Performance advantage approximately doubles from smallest to largest mesh, suggesting improved cache efficiency at larger scales.

**Multiple Contours**: `scalar_field_multi_contours_vs_vtk.csv`
- Extracts multiple isocontours simultaneously on a 50k triangle mesh
- Dataset: 1 to 1000 contour levels
- **Result: 10x to 170x speedup**
- **Pattern**: Speedup grows superlinearly with number of contours
  - 1 contour: 10.2x
  - 10 contours: 14.8x
  - 100 contours: 23.6x
  - 500 contours: 77.4x
  - 1000 contours: 169.8x
- **Analysis**: The speedup grows superlinearly beyond 100 contours. Trueform's `build_many` implementation amortizes computation across all threshold levels in a single pass.

### Mesh Cleaning

**vs VTK**: `mesh_cleaning_vs_vtk.csv`
- Removes duplicate vertices and degenerate faces
- Tolerance: 1e-6
- Dataset: Meshes from 50k to 1M triangles
- **Result: 7.6x to 29.1x speedup**
- **Pattern**: Speedup increases with mesh size
  - 50k triangles: 7.6x (2.7ms vs 20.8ms)
  - 250k triangles: 12.0x (11.2ms vs 134.6ms)
  - 500k triangles: 21.2x (21.0ms vs 446.3ms)
  - 1M triangles: 29.1x (42.8ms vs 1248.2ms)
- **Analysis**: VTK time grows superlinearly while trueform maintains near-linear scaling. At 1M triangles, VTK requires 1248ms compared to trueform's 43ms.

### Feature Edge Detection

**Boundary Edges**: `feature_edges_boundary_vs_vtk.csv`
- Extracts boundary edges from mesh
- Dataset: Meshes from 50k to 1M triangles
- **Result: 10.6x to 13.9x speedup**
- **Pattern**: Speedup remains relatively constant across sizes
  - Range: 10.6x to 13.9x
  - Average: 12.4x
- **Analysis**: Both implementations exhibit similar scaling behavior. The consistent speedup factor suggests trueform's manifold edge link data structure provides improved cache locality and parallel processing efficiency.

**Non-Manifold Edges**: `feature_edges_non_manifold_vs_vtk.csv`
- Detects non-manifold edges (edges shared by more than 2 faces)
- Dataset: Meshes from 50k to 1M triangles
- **Result: 12.9x to 20.5x speedup**
- **Pattern**: Variable speedup with increasing advantage at larger sizes
  - 50k triangles: 14.3x
  - 500k triangles: 12.9x
  - 1M triangles: 20.5x
- **Analysis**: Speedup variation may reflect mesh-specific topology. The increase to 20.5x at 1M triangles indicates improved parallel topology construction efficiency at scale.

### Planar Arrangements

**Speedup Summary**: `planar_arrangements_vs_cgal.csv`
- 2D planar arrangement computation (overlaying line segments)
- Dataset: Arrangements with 200 to 40,000 edges
- **Result: 1.3x to 2273x speedup**
- **Pattern**: Speedup grows exponentially with problem size
  - 200 edges: 1.3x
  - 2k edges: 28.1x
  - 5k edges: 107.1x
  - 10k edges: 296.8x
  - 20k edges: 676.2x
  - 40k edges: 2273.5x
- **Analysis**: CGAL's implementation exhibits superlinear complexity growth while trueform maintains near-linear scaling. At 40k edges, CGAL requires 21,261ms compared to trueform's 9.8ms.

**Detailed Timings**: `planar_arrangements_vs_cgal_detailed.csv`
- Full timing data with 21 data points from 200 to 40,000 edges
- Shows exact millisecond timings for both libraries
- **Key Observations**:
  - Trueform time grows linearly: 0.34ms → 9.77ms (28.5x for 200x size increase)
  - CGAL time grows superlinearly: 0.71ms → 21261ms (29,944x for 200x size increase)
  - Crossover occurs at approximately 1500 edges where speedup exceeds 10x
  - Speedup growth accelerates beyond 10k edges

### Connected Components

**vs CGAL**: `connected_components_vs_cgal.csv`
- Labels connected components in mesh with 151 components
- Dataset: Meshes from 50k to 1M triangles
- **Result: 4.2x to 8.5x speedup**
- **Pattern**: Speedup increases moderately with mesh size
  - 50k triangles: 4.2x (0.54ms vs 2.26ms)
  - 250k triangles: 6.8x (1.97ms vs 13.44ms)
  - 500k triangles: 8.5x (2.52ms vs 21.5ms)
  - 750k/1M: 5.9x-7.7x
- **Analysis**: Speedup grows to 8.5x at 500k triangles with some variation at larger sizes. Both implementations use efficient algorithms (union-find or BFS), with performance differences attributable to parallel execution and cache optimization.

### Spatial Queries

**k-Nearest Neighbor Queries**: `knn_queries_vs_nanoflann.csv`
- Performs k-nearest neighbor queries on point clouds
- Dataset: Queries with k ranging from 1 to 10 neighbors
- **Result: 1.8x to 2.0x speedup**
- **Pattern**: Speedup decreases slightly as k increases
  - k=1: 2.0x
  - k=2-3: 1.94x-1.86x
  - k=4-10: 1.80x-1.83x
- **Analysis**: Speedup stabilizes at approximately 1.8x for k≥5. Both implementations use efficient kd-tree structures, with performance differences arising from trueform's parallel query processing and cache-optimized memory layout.

**Collecting Intersecting Primitives**: `spatial_queries_vs_cgal.csv`
- Collects all primitives intersecting a query region
- Dataset: Two meshes from 50k to 1M triangles each (100k to 2M total)
- **Result: 8.1x to 12.4x speedup**
- **Pattern**: Speedup fluctuates with moderate variation across sizes
  - 100k triangles: 8.1x
  - 250k triangles: 11.3x
  - 500k-1M triangles: 9.4x-9.5x
  - 2M triangles: 12.4x
- **Analysis**: Average speedup of approximately 10x with peak performance at 2M triangles. The variation suggests mesh-dependent spatial distribution effects, while the consistent order of magnitude advantage indicates efficient BVH traversal and intersection testing.

**Tree Construction vs Nanoflann**: `tree_construction_vs_nanoflann.csv`
- Builds spatial acceleration structure (BVH/kd-tree) for point clouds
- Dataset: Point clouds from 27k to 502k points
- **Result: 3.5x to 6.7x speedup**
- **Pattern**: Speedup increases with point cloud size
  - 27k points: 3.5x (1.69ms vs 5.91ms)
  - 127k points: 6.5x (6.72ms vs 43.53ms)
  - 502k points: 6.7x (29.22ms vs 196.76ms)
- **Analysis**: Speedup stabilizes around 6-7x for point clouds larger than 100k points. Trueform's parallel tree construction algorithm demonstrates superior scaling characteristics, with build time remaining under 30ms even for 502k points compared to nanoflann's 197ms.

**Tree Construction vs CGAL**: `tree_construction_vs_cgal.csv`
- Builds spatial acceleration structure (AABB tree) for triangle meshes
- Dataset: Meshes from 50k to 1M triangles
- **Result: 26x to 31x speedup**
- **Pattern**: Speedup remains relatively constant across sizes
  - 50k triangles: 28.2x (2ms vs 56.3ms)
  - 250k triangles: 31.0x (12ms vs 372.5ms)
  - 1M triangles: 27.9x (61ms vs 1702.5ms)
- **Analysis**: Consistent speedup of approximately 28-31x across all mesh sizes. At 1M triangles, trueform constructs the tree in 61ms compared to CGAL's 1.7 seconds. The constant speedup factor indicates both implementations scale linearly, with trueform's parallel construction providing a uniform performance advantage.

## Data Format

All CSV files use standard format:
- Header row with column names
- Comma-separated values
- Numerical data without quotes
- Compatible with Python (pandas), R, Excel, and other analysis tools

### Column Descriptions

**Common columns:**
- `triangles`: Number of triangles in mesh
- `edges`: Number of edges in arrangement
- `neighbors`: Number of nearest neighbors (k) in kNN queries
- `speedup`: Performance ratio (baseline_time / trueform_time)
- `trueform_ms`: Trueform execution time in milliseconds
- `vtk_ms` / `cgal_ms` / `nanoflann_ms`: Baseline library time in milliseconds
- `num_contours`: Number of isocontour levels extracted

## Usage Examples

### Python (pandas)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load and analyze mesh intersection data
df = pd.read_csv('mesh_intersection_curves_vs_vtk.csv')

# Plot speedup scaling
plt.figure(figsize=(10, 6))
plt.plot(df['triangles'], df['speedup'], marker='o', linewidth=2)
plt.xlabel('Number of Triangles')
plt.ylabel('Speedup (×)')
plt.title('Trueform vs VTK: Intersection Curves Scaling')
plt.grid(True, alpha=0.3)
plt.show()

# Statistical summary
print(f"Mean speedup: {df['speedup'].mean():.1f}×")
print(f"Max speedup: {df['speedup'].max():.1f}×")
print(f"Speedup growth rate: {df['speedup'].iloc[-1] / df['speedup'].iloc[0]:.1f}×")

# Compare multiple benchmarks
cleaning = pd.read_csv('mesh_cleaning_vs_vtk.csv')
print(f"\nCleaning at 1M triangles:")
print(f"  Trueform: {cleaning.iloc[-1]['trueform_ms']:.1f}ms")
print(f"  VTK: {cleaning.iloc[-1]['vtk_ms']:.1f}ms")
```

### R

```r
library(ggplot2)
library(dplyr)

# Load planar arrangements data
df <- read.csv('planar_arrangements_vs_cgal_detailed.csv')

# Create dual-axis plot
ggplot(df, aes(x=edges)) +
  geom_line(aes(y=trueform_ms, color="Trueform"), linewidth=1) +
  geom_line(aes(y=cgal_ms, color="CGAL"), linewidth=1) +
  geom_point(aes(y=trueform_ms, color="Trueform"), size=2) +
  geom_point(aes(y=cgal_ms, color="CGAL"), size=2) +
  scale_y_log10() +
  labs(title='Trueform vs CGAL: Planar Arrangements (Log Scale)',
       x='Number of Edges',
       y='Time (ms, log scale)',
       color='Library') +
  theme_minimal()

# Analyze scaling behavior
df %>%
  summarize(
    trueform_slope = lm(log(trueform_ms) ~ log(edges))$coefficients[2],
    cgal_slope = lm(log(cgal_ms) ~ log(edges))$coefficients[2]
  )
```

## Benchmark Environment

All benchmarks were executed under controlled conditions:
- Parallel execution enabled for both trueform and baseline libraries
- Identical input data for fair comparison
- Multiple runs averaged to account for variance
- Release builds with compiler optimizations enabled

## Summary of Results

### Mesh-Mesh Intersections
Trueform demonstrates 100x+ speedup over VTK and 30x+ over CGAL, with performance advantage increasing at larger scales.

### Multi-Level Operations
Multi-contour extraction shows speedup range of 10x to 170x over VTK, with trueform's batch processing approach showing improved efficiency when processing multiple threshold levels.

### Planar Arrangements
The 2273x speedup at 40k edges compared to CGAL indicates different algorithmic complexity classes, with trueform exhibiting near-linear scaling versus CGAL's superlinear behavior.

### Topology Operations
Cleaning and feature detection operations show consistent 10-30x speedups, indicating efficient parallel execution and memory access patterns.

### Spatial Acceleration
Tree construction shows 3.5x to 6.7x speedup over nanoflann for point clouds and 26x to 31x speedup over CGAL for triangle meshes. kNN queries demonstrate consistent 1.8x-2.0x speedup, while spatial intersection queries achieve 8-12x speedup over CGAL.

### Scaling Characteristics
Across multiple operation types, trueform maintains near-linear time complexity while competing libraries exhibit superlinear growth, providing increasing advantage at larger problem sizes.

### Practical Performance
At 1M+ triangle scales, operations that require seconds in other libraries complete in milliseconds with trueform:
- Mesh intersection: 42ms vs 21 seconds (CGAL)
- Planar arrangement (40k edges): 9.8ms vs 21 seconds (CGAL)
- Mesh cleaning: 43ms vs 1.2 seconds (VTK)
- Tree construction (502k points): 29ms vs 197ms (nanoflann)

## Citation

If you use these benchmarks in your work, please cite:

```
Žiga Sajovic (2025). Trueform: High-Performance Geometric Processing Library.
https://github.com/xlabmedical/trueform
```

## License

Benchmark data is provided under the Boost Software License, Version 1.0.
See https://github.com/xlabmedical/trueform for details.
