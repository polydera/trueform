/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/vtk/core/make_points.hpp>
#include <vtkType.h>

namespace tf::vtk {

/// @brief Edges view type (tf::edges wrapping blocked_buffer<vtkIdType, 2>).
using edges_t =
    decltype(tf::make_edges(std::declval<const tf::blocked_buffer<vtkIdType, 2>>()));

/// @brief Segments view type (edges + points).
using segments_t =
    decltype(tf::make_segments(std::declval<edges_t>(), std::declval<points_t>()));

} // namespace tf::vtk
