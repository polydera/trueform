/**
 * Benchmark: Mesh–mesh distance (Coal)
 *
 * - Start from a triangle mesh (TrueForm polygons)
 * - Copy points + triangle indices into a Coal BVH model
 * - Build the BVH once
 * - Two instances of the same mesh: fixed and moving
 * - Each sample:
 *     - choose a random pivot on the mesh
 *     - build a random rigid transform at that pivot (TrueForm)
 *     - convert it to a Coal/Eigen transform for the moving mesh
 *     - run one mesh–mesh distance query with Coal
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run Coal polygons to polygons closest-point benchmark
 *
 * Outputs CSV with columns: bv,polygons,polygons,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_polygons_to_polygons_closest_point_coal_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
