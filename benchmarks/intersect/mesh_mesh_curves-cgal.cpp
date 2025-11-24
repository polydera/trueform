/**
 * Benchmark: Mesh-mesh intersection curves with CGAL
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "mesh_mesh_curves-cgal.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_mesh_mesh_curves_cgal_benchmark(
      benchmark::BENCHMARK_MESHES,
      100, // n_samples
      std::cout);
}
