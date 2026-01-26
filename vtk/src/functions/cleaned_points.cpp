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
#include <trueform/vtk/functions/cleaned_points.hpp>
#include <trueform/clean.hpp>
#include <trueform/vtk/core/make_points.hpp>
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <vtkPoints.h>

namespace tf::vtk {

auto cleaned_points(vtkPoints *input, float tolerance)
    -> vtkSmartPointer<vtkPoints> {
  if (!input || input->GetNumberOfPoints() == 0) {
    return nullptr;
  }

  auto points = make_points(input);
  auto cleaned = tf::cleaned<vtkIdType>(points, tolerance);

  return make_vtk_points(std::move(cleaned));
}

} // namespace tf::vtk
