/**
 * Point normals benchmark with VTK - Implementation
 *
 * Uses vtkPolyDataNormals with only point normal computation enabled.
 * Consistency checking, splitting, and auto-orient are disabled for
 * fair comparison with other libraries.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "point_normals-vtk.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <vtkPolyDataNormals.h>
#include <vtkSmartPointer.h>

namespace benchmark {

int run_point_normals_vtk_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);

    // Convert to VTK polydata
    auto polydata = benchmark::vtk::to_vtk_polydata(mesh);

    auto time = benchmark::min_time_of(
        [&]() {
          auto normals = vtkSmartPointer<vtkPolyDataNormals>::New();
          normals->SetInputData(polydata);
          // Only compute point normals, disable extra processing
          normals->ComputePointNormalsOn();
          normals->ComputeCellNormalsOff();
          normals->ConsistencyOff();
          normals->SplittingOff();
          normals->AutoOrientNormalsOff();
          normals->Update();
          benchmark::do_not_optimize(normals->GetOutput());
        },
        n_samples);

    out << mesh.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
