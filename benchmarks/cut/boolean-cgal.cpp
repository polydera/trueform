/**
 * Benchmark: Boolean operations with CGAL
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boolean-cgal.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_boolean_cgal_benchmark(benchmark::BENCHMARK_MESHES,
                                               10, // n_samples
                                               std::cout);
}
