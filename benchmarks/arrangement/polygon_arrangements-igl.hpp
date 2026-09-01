/**
 * Polygon arrangements benchmark with libigl
 *
 * Measures time to compute polygon arrangements using
 * igl::copyleft::cgal::remesh_self_intersections.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

int run_polygon_arrangements_igl_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
