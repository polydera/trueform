/**
 * Benchmark: Polygons tree building with CGAL
 *
 * Measures time to build spatial acceleration structure (AABB tree)
 * on triangle meshes of varying sizes using CGAL's AABB tree.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run CGAL tree building benchmark
 *
 * Outputs CSV with columns: polygons,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_polygons_build_tree_cgal_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
