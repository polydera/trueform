/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/vtk/core/make_paths.hpp>
#include <trueform/vtk/core/make_points.hpp>
#include <vtkPolyData.h>

namespace tf::vtk {

using curves_t = decltype(tf::make_curves(
    std::declval<paths_t>(), std::declval<points_t>()));

auto make_curves(vtkPolyData *poly) -> curves_t;

} // namespace tf::vtk
