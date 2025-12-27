/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (www.trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#include <trueform/vtk/core/make_curves.hpp>

namespace tf::vtk {

auto make_curves(vtkPolyData *poly) -> curves_t {
  return tf::make_curves(make_paths(poly->GetLines()), make_points(poly));
}

} // namespace tf::vtk
