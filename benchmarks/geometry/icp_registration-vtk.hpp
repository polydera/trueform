/**
 * ICP registration benchmark with VTK
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <ostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run ICP registration benchmark using VTK.
 *
 * For each mesh:
 * - Creates target by Taubin smoothing (200 iterations)
 * - Creates source by applying random rotation to original
 * - Runs vtkIterativeClosestPointTransform with StartByMatchingCentroids
 *
 * @param mesh_paths Paths to test meshes
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 for success)
 */
int run_icp_registration_vtk_benchmark(const std::vector<std::string> &mesh_paths,
                                       int n_samples, std::ostream &out);

} // namespace benchmark
