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

namespace tf::vtk {

class polydata;

/// @brief Compute point normals and set them on the polydata.
/// @param input The polydata (must contain 3D polygons).
/// @note If cell normals are not present, they will be computed first.
auto compute_point_normals(polydata *input) -> void;

} // namespace tf::vtk
