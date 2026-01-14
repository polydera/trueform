/**
 * Point normals benchmark with libigl
 *
 * Measures time to compute vertex normals for triangle meshes
 * using libigl's per_vertex_normals.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run libigl point normals benchmark
 *
 * Outputs CSV with columns: polygons,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_point_normals_igl_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
