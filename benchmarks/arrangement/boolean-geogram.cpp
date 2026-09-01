/**
 * Benchmark: Boolean operations with Geogram
 *
 * Standalone executable entry point.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boolean-geogram.hpp"
#include "test_meshes.hpp"

#include <geogram/basic/common.h>

#include <iostream>

int main() {
  GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);
  return benchmark::run_boolean_geogram_benchmark(benchmark::BENCHMARK_MESHES,
                                                  5, // n_samples
                                                  std::cout);
}
