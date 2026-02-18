/**
 * Benchmark: Isotropic remeshing with CGAL
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "isotropic_remeshing-cgal.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_isotropic_remeshing_cgal_benchmark(
      benchmark::BENCHMARK_MESHES,
      2, // n_samples
      std::cout);
}
