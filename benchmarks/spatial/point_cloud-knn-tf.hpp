/**
 * Point cloud kNN queries benchmark with TrueForm
 *
 * Measures time to perform k-nearest neighbor queries on point clouds
 * of varying sizes using TrueForm's tf::aabb_tree.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run TrueForm point cloud kNN queries benchmark
 *
 * Outputs CSV with columns: points,k,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_point_cloud_knn_tf_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
