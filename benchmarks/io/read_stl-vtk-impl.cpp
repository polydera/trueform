/**
 * STL reading benchmark with VTK - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "read_stl-vtk.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>

namespace benchmark {

int run_read_stl_vtk_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    // First read to get polygon count
    auto mesh_for_count = tf::read_stl<int>(path);
    const auto n_polygons = mesh_for_count.faces().size();

    auto time = benchmark::min_time_of(
        [&]() {
          auto reader = vtkSmartPointer<vtkSTLReader>::New();
          reader->SetFileName(path.c_str());
          reader->Update();
          benchmark::do_not_optimize(reader->GetOutput());
        },
        n_samples);

    out << n_polygons << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
