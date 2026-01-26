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

class vtkPolyData;

namespace tf::vtk {

class polydata;

/// @brief Compute cell normals and set them on the vtkPolyData.
/// @param input The polydata (must contain 3D polygons).
auto compute_cell_normals(vtkPolyData *input) -> void;

} // namespace tf::vtk
