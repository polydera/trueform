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
#include <utility>

class vtkMatrix4x4;

namespace tf::vtk {

class polydata;

/// @brief Compute mean edge length of a mesh.
/// @param input The polydata to measure.
/// @return Mean edge length.
auto mean_edge_length(polydata *input) -> float;

/// @brief Compute mean edge length of a mesh with a transformation frame.
/// @param input The polydata and its transformation matrix.
/// @return Mean edge length in the transformed coordinate space.
auto mean_edge_length(std::pair<polydata *, vtkMatrix4x4 *> input) -> float;

} // namespace tf::vtk
