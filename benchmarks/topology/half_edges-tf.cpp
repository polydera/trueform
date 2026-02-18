/**
 * Benchmark: Half-edge construction with TrueForm
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "half_edges-tf.hpp"
#include "test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_half_edges_tf_benchmark(benchmark::BENCHMARK_MESHES,
                                                10, std::cout);
}
