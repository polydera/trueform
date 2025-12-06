/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/core/make_points.hpp>
#include <vtkPoints.h>
#include <vtkPolyData.h>

namespace tf::vtk {

auto make_points(vtkPoints *points) -> points_t {
  if (!points) {
    return tf::make_points<3>(tf::make_range(static_cast<float *>(nullptr), 0));
  }
  auto *ptr = static_cast<float *>(points->GetData()->GetVoidPointer(0));
  return tf::make_points<3>(
      tf::make_range(ptr, 3 * points->GetNumberOfPoints()));
}

auto make_points(vtkPolyData *poly) -> points_t {
  if (!poly) {
    return tf::make_points<3>(tf::make_range(static_cast<float *>(nullptr), 0));
  }
  return make_points(poly->GetPoints());
}

} // namespace tf::vtk
