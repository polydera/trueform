/**
 * Benchmark: Point cloud tree building with nanoflann
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "point_cloud-build_tree-nanoflann.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_point_cloud_build_tree_nanoflann_benchmark(
      benchmark::BENCHMARK_MESHES,
      10, // n_samples
      std::cout);
}
