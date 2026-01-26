/**
 * Benchmark: Polygons to polygons closest-point FCL
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "polygons_to_polygons-closest_point-fcl.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_polygons_to_polygons_closest_point_fcl_benchmark(
      benchmark::BENCHMARK_MESHES,
      1000, // n_samples
      std::cout);
}
