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
#include <vtkSmartPointer.h>

class vtkPoints;

namespace tf::vtk {

/// @brief Creates vtkPoints from a points_buffer (copies data).
/// @param points Trueform points buffer.
/// @return A new vtkPoints object with copied data.
auto make_vtk_points(const tf::points_buffer<float, 3> &points)
    -> vtkSmartPointer<vtkPoints>;

/// @brief Creates vtkPoints from a points_buffer (moves data).
/// @param points Trueform points buffer (consumed).
/// @return A new vtkPoints object with transferred ownership.
auto make_vtk_points(tf::points_buffer<float, 3> &&points)
    -> vtkSmartPointer<vtkPoints>;

} // namespace tf::vtk
