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
#include <trueform/vtk/core/make_polygons.hpp>

namespace tf::vtk {

auto make_polygons(vtkPolyData *poly) -> polygons_t {
  return tf::make_polygons(make_polys(poly ? poly->GetPolys() : nullptr),
                           make_points(poly));
}

template auto make_polygons<3>(vtkPolyData *poly) -> polygons_sized_t<3>;

} // namespace tf::vtk
