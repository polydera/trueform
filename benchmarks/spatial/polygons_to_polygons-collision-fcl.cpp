/**
 * Benchmark: Polygons to polygons collision FCL
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "polygons_to_polygons-collision-fcl.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_polygons_to_polygons_collision_fcl_benchmark(
      benchmark::BENCHMARK_MESHES,
      1000, // n_samples
      std::cout);
}
