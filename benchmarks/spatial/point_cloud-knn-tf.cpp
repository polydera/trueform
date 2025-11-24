/**
 * Benchmark: Point cloud kNN queries with TrueForm
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "point_cloud-knn-tf.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_point_cloud_knn_tf_benchmark(
      benchmark::BENCHMARK_MESHES,
      100, // n_samples
      std::cout);
}
