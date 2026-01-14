/**
 * Benchmark: Principal curvatures with libigl
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "principal_curvatures-igl.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_principal_curvatures_igl_benchmark(
      benchmark::BENCHMARK_MESHES,
      10, // n_samples
      std::cout);
}
