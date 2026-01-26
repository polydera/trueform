/**
 * Benchmark: Mesh-mesh collision (FCL)
 *
 * - Start from a triangle mesh (TrueForm polygons)
 * - Copy points + triangle indices into an FCL BVH model
 * - Build the BVH once
 * - Two instances of the same mesh: fixed and moving
 * - Each sample:
 *     - choose a random pivot on the mesh
 *     - build a random rigid transform at that pivot (TrueForm)
 *     - convert it to an FCL/Eigen transform for the moving mesh
 *     - run one mesh-mesh collision query with FCL
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run FCL polygons to polygons collision benchmark
 *
 * Outputs CSV with columns: bv,polygons,polygons,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_polygons_to_polygons_collision_fcl_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
