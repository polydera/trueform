/**
 * Mesh-mesh intersection curves benchmark with VTK
 *
 * Measures time to compute intersection curves between two meshes
 * using VTK's vtkIntersectionPolyDataFilter.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run VTK mesh-mesh intersection curves benchmark
 *
 * Outputs CSV with columns: polygons0,polygons1,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_mesh_mesh_curves_vtk_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
