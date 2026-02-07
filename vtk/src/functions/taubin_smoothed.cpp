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
#include <trueform/geometry/taubin_smoothed.hpp>
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/taubin_smoothed.hpp>

namespace tf::vtk {

auto taubin_smoothed(polydata *input, std::size_t iterations, float lambda,
                     float kpb) -> vtkSmartPointer<vtkPoints> {
  if (!input || input->GetNumberOfPoints() == 0) {
    return nullptr;
  }

  auto points = input->points();
  const auto &vlink = input->vertex_link();

  auto result = tf::taubin_smoothed(points | tf::tag(vlink), iterations, lambda, kpb);

  return make_vtk_points(result);
}

} // namespace tf::vtk
