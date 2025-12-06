/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/vtk/core/make_lines.hpp>
#include <trueform/vtk/core/make_points.hpp>
#include <vtkPolyData.h>

namespace tf::vtk {

/// @brief Segments view type (edge pairs).
using segments_t = decltype(tf::make_segments(
    std::declval<lines_t<2>>(), std::declval<points_t>()));

/// @brief Creates a segments view from vtkPolyData lines (zero-copy).
/// @param poly VTK poly data object, may be nullptr.
/// @return A tf::segments view over the underlying lines data.
auto make_segments(vtkPolyData *poly) -> segments_t;

} // namespace tf::vtk
