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

class vtkMatrix4x4;

namespace tf::vtk {

/// @brief Creates a trueform frame from a vtkMatrix4x4.
/// @param matrix VTK 4x4 transformation matrix.
/// @return A tf::frame with both transformation and inverse computed.
auto make_frame(vtkMatrix4x4 *matrix) -> tf::frame<double, 3>;

/// @brief Fills a trueform frame from a vtkMatrix4x4.
/// @param frame The frame to fill.
/// @param matrix VTK 4x4 transformation matrix.
template <typename T>
auto fill_frame(tf::frame<T, 3> &frame, vtkMatrix4x4 *matrix) -> void;

} // namespace tf::vtk
