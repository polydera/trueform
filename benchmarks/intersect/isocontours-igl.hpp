/**
 * Isocontours benchmark with libigl
 *
 * Measures time to extract isocontours from a scalar field on a mesh
 * using libigl's igl::isolines.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run libigl isocontours benchmark
 *
 * Outputs CSV with columns: polygons,n_cuts,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_isocontours_igl_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
