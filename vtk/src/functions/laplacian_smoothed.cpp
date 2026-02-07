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
#include <trueform/geometry/laplacian_smoothed.hpp>
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/laplacian_smoothed.hpp>

namespace tf::vtk {

auto laplacian_smoothed(polydata *input, std::size_t iterations, float lambda)
    -> vtkSmartPointer<vtkPoints> {
  if (!input || input->GetNumberOfPoints() == 0) {
    return nullptr;
  }

  auto points = input->points();
  const auto &vlink = input->vertex_link();

  auto result = tf::laplacian_smoothed(points | tf::tag(vlink), iterations, lambda);

  return make_vtk_points(result);
}

} // namespace tf::vtk
