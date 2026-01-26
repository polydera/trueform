/**
 * Embedded self-intersection curves benchmark with libigl
 *
 * Measures time to compute self-intersection resolution on a mesh made by
 * concatenating two overlapping copies using igl::copyleft::cgal::remesh_self_intersections.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run libigl embedded self-intersection curves benchmark
 *
 * Outputs CSV with columns: polygons0,polygons1,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_embedded_self_intersection_curves_igl_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
