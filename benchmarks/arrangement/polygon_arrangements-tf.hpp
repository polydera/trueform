/**
 * Polygon arrangements benchmark with TrueForm
 *
 * Measures time to compute polygon arrangements on a mesh made by
 * concatenating two overlapping copies of the same mesh.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace benchmark {

int run_polygon_arrangements_tf_benchmark(
    const std::vector<std::string>& mesh_paths,
    int n_samples,
    std::ostream& out
);

} // namespace benchmark
