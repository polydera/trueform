/**
 * STL reading benchmark with VTK
 *
 * Measures time to read STL files using VTK's vtkSTLReader.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run VTK STL reading benchmark
 *
 * Outputs CSV with columns: polygons,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_read_stl_vtk_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
