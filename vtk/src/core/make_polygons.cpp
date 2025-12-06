/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/core/make_polygons.hpp>

namespace tf::vtk {

auto make_polygons(vtkPolyData *poly) -> dynamic_polygons_t {
  return tf::make_polygons(make_polys(poly ? poly->GetPolys() : nullptr),
                           make_points(poly));
}

template auto make_polygons<3>(vtkPolyData *poly) -> polygons_t<3>;

} // namespace tf::vtk
