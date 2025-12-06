/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/vtk/core/make_segments.hpp>

namespace tf::vtk {

auto make_segments(vtkPolyData *poly) -> segments_t {
  return tf::make_segments(make_lines<2>(poly ? poly->GetLines() : nullptr),
                           make_points(poly));
}

} // namespace tf::vtk
