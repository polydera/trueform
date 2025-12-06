/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
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
