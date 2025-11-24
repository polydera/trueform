/**
 * Benchmark: Boundary paths with libigl
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boundary_paths-igl.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_boundary_paths_igl_benchmark(
      benchmark::BENCHMARK_MESHES,
      10, // n_samples
      std::cout);
}
