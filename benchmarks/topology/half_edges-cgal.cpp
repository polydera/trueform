/**
 * Benchmark: Half-edge construction with CGAL
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "half_edges-cgal.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_half_edges_cgal_benchmark(benchmark::BENCHMARK_MESHES,
                                                  10, std::cout);
}
