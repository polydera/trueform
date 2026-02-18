/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#include <trueform/core/mean_edge_length.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/mean_edge_length.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto mean_edge_length(polydata *input) -> float {
  if (!input || input->GetNumberOfPolys() == 0) {
    return 0.f;
  }

  return tf::mean_edge_length(input->polygons());
}

auto mean_edge_length(std::pair<polydata *, vtkMatrix4x4 *> input) -> float {
  auto [mesh, matrix] = input;
  if (!mesh || mesh->GetNumberOfPolys() == 0) {
    return 0.f;
  }

  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());

  return tf::mean_edge_length(mesh->polygons() | tf::tag(frame));
}

} // namespace tf::vtk
