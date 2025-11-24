/**
 * Boolean operations benchmark with libigl
 *
 * Measures time to compute boolean union between two meshes
 * using libigl's mesh_boolean.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run libigl boolean operations benchmark
 *
 * Outputs CSV with columns: polygons0,polygons1,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_boolean_igl_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
