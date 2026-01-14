/**
 * Benchmark: Point normals with libigl
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "point_normals-igl.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_point_normals_igl_benchmark(
      benchmark::BENCHMARK_MESHES,
      10, // n_samples
      std::cout);
}
