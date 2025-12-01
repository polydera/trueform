/**
 * Shared benchmark test meshes
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#pragma once

#include <string>
#include <vector>

namespace benchmark {

/**
 * Standard set of test meshes used across all benchmarks.
 * Provides consistent test data for spatial, topology, intersect, cut, etc.
 */
inline const std::vector<std::string> BENCHMARK_MESHES = {
    "benchmarks/data/dragon-50k.stl",
    "benchmarks/data/dragon-125k.stl",
    "benchmarks/data/dragon-250k.stl",
    "benchmarks/data/dragon-500k.stl",
    "benchmarks/data/dragon-750k.stl",
    "benchmarks/data/dragon-1M.stl"
};

} // namespace benchmark
