/**
 * Mesh-mesh intersection curves benchmark with TrueForm
 *
 * Measures time to compute intersection curves between two meshes
 * using TrueForm's make_intersection_curves.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run TrueForm mesh-mesh intersection curves benchmark
 *
 * Outputs CSV with columns: polygons0,polygons1,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_mesh_mesh_curves_tf_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
