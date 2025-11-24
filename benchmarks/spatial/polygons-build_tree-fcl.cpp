/**
 * Benchmark: Polygons tree building FCL
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "polygons-build_tree-fcl.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_polygons_build_tree_fcl_benchmark(
      benchmark::BENCHMARK_MESHES,
      10, // n_samples
      std::cout);
}
