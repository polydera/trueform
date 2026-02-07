/**
 * ICP registration benchmark with libigl - Entry point
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "icp_registration-igl.hpp"
#include "../common/test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_icp_registration_igl_benchmark(benchmark::BENCHMARK_MESHES,
                                                       30, std::cout);
}
