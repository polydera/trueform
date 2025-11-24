/**
 * Boolean operations benchmark with CGAL
 *
 * Measures time to compute boolean union between two meshes
 * using CGAL's PMP::corefine_and_compute_union.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run CGAL boolean operations benchmark
 *
 * Outputs CSV with columns: polygons0,polygons1,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_boolean_cgal_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
