/**
 * Benchmark: Embedded isocurves with VTK
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "embedded_isocurves-vtk.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_embedded_isocurves_vtk_benchmark(
      benchmark::BENCHMARK_MESHES,
      10, // n_samples
      std::cout);
}
