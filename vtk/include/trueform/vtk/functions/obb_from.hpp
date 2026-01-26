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

class vtkPolyData;

namespace tf::vtk {

/// @brief Compute oriented bounding box from polydata points.
/// @param input The vtkPolyData.
/// @return The oriented bounding box.
auto obb_from(vtkPolyData *input) -> tf::obb<float, 3>;

} // namespace tf::vtk
