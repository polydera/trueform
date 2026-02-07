/**
 * ICP registration benchmark with VTK - Entry point
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "icp_registration-vtk.hpp"
#include "../common/test_meshes.hpp"
#include <iostream>

int main() {
  return benchmark::run_icp_registration_vtk_benchmark(benchmark::BENCHMARK_MESHES,
                                                       30, std::cout);
}
