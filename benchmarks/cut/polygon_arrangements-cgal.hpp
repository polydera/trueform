/**
 * Polygon arrangements benchmark with CGAL
 *
 * Measures time to compute polygon arrangements using
 * CGAL::Polygon_mesh_processing::autorefine_triangle_soup.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

int run_polygon_arrangements_cgal_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
