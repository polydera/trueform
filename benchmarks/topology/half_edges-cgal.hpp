/**
 * Half-edge construction benchmark with CGAL
 *
 * Measures time to build a CGAL Surface_mesh (half-edge data structure)
 * from vertex and face data.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run CGAL half-edge construction benchmark
 *
 * Outputs CSV with columns: polygons,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_half_edges_cgal_benchmark(const std::vector<std::string> &mesh_paths,
                                  int n_samples, std::ostream &out);

} // namespace benchmark
