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
#pragma once
#include <trueform/core.hpp>

class vtkPoints;
class vtkPolyData;

namespace tf::vtk {

/// @brief Points view type for VTK float data.
using points_t = decltype(tf::make_points<3>(
    tf::make_range(static_cast<float *>(nullptr), std::size_t{0})));

/// @brief Creates a points view from vtkPoints (zero-copy).
/// @param points VTK points object, may be nullptr.
/// @return A tf::points view over the underlying data, or empty if nullptr.
auto make_points(vtkPoints *points) -> points_t;

/// @brief Creates a points view from vtkPolyData (zero-copy).
/// @param poly VTK poly data object, may be nullptr.
/// @return A tf::points view over the underlying data, or empty if nullptr.
auto make_points(vtkPolyData *poly) -> points_t;

} // namespace tf::vtk
