/**
 * Benchmark: Collision queries with trueform
 *
 * Measures time to perform mesh-mesh collision detection queries
 * using TrueForm's tf::intersects.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run TrueForm polygons to polygons collision benchmark
 *
 * Outputs CSV with columns: bv,polygons,polygons,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_polygons_to_polygons_collision_tf_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
