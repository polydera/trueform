/**
 * Boundary paths benchmark with VTK - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boundary_paths-vtk.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <vtkFeatureEdges.h>
#include <vtkSmartPointer.h>

namespace benchmark {

int run_boundary_paths_vtk_benchmark(const std::vector<std::string> &mesh_paths,
                                     int n_samples, std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);

    // Convert to VTK polydata
    auto polydata = benchmark::vtk::to_vtk_polydata(r_polygons);

    auto time = benchmark::min_time_of(
        [&]() {
          auto featureEdges = vtkSmartPointer<vtkFeatureEdges>::New();
          featureEdges->SetInputData(polydata);
          featureEdges->BoundaryEdgesOn();
          featureEdges->FeatureEdgesOff();
          featureEdges->ManifoldEdgesOff();
          featureEdges->NonManifoldEdgesOff();
          featureEdges->Update();
          benchmark::do_not_optimize(featureEdges->GetOutput());
        },
        n_samples);

    out << r_polygons.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
