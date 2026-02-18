/**
 * Isotropic remeshing benchmark with CGAL
 *
 * Measures time to isotropic remesh triangle meshes
 * using CGAL's PMP::isotropic_remeshing.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Run CGAL isotropic remeshing benchmark
 *
 * Outputs CSV with columns: polygons,multiplier,time_ms
 *
 * @param mesh_paths Vector of STL file paths to benchmark
 * @param n_samples Number of timing samples per mesh
 * @param out Output stream for CSV results
 * @return Exit code (0 = success)
 */
int run_isotropic_remeshing_cgal_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
