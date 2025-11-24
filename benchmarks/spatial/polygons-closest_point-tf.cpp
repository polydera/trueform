/**
 * Benchmark: Polygons closest-point TrueForm
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "polygons-closest_point-tf.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_polygons_closest_point_tf_benchmark(
      benchmark::BENCHMARK_MESHES,
      100, // n_samples
      std::cout);
}
