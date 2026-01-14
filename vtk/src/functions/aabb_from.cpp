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
#include <trueform/vtk/core/make_points.hpp>
#include <trueform/vtk/functions/aabb_from.hpp>
#include <vtkPolyData.h>

namespace tf::vtk {

auto aabb_from(vtkPolyData *input) -> tf::aabb<float, 3> {
  auto points = make_points(input);
  return tf::aabb_from(points);
}

} // namespace tf::vtk
