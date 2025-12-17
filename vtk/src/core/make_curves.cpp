/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/core/make_curves.hpp>

namespace tf::vtk {

auto make_curves(vtkPolyData *poly) -> curves_t {
  return tf::make_curves(make_paths(poly->GetLines()), make_points(poly));
}

} // namespace tf::vtk
