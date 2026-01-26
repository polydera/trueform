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
#include <trueform/vtk/core/polydata.hpp>
#include <utility>

class vtkMatrix4x4;

namespace tf::vtk {

/// @brief Check if two polydata meshes intersect.
/// @param input0 The first polydata.
/// @param input1 The second polydata.
/// @return True if the meshes intersect, false otherwise.
auto intersects(polydata *input0, polydata *input1) -> bool;

/// @brief Check if two polydata meshes intersect, with transform on first.
/// @param input0 The first polydata with transform.
/// @param input1 The second polydata.
/// @return True if the meshes intersect, false otherwise.
auto intersects(std::pair<polydata *, vtkMatrix4x4 *> input0, polydata *input1)
    -> bool;

/// @brief Check if two polydata meshes intersect, with transform on second.
/// @param input0 The first polydata.
/// @param input1 The second polydata with transform.
/// @return True if the meshes intersect, false otherwise.
auto intersects(polydata *input0, std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> bool;

/// @brief Check if two polydata meshes intersect, both with transforms.
/// @param input0 The first polydata with transform.
/// @param input1 The second polydata with transform.
/// @return True if the meshes intersect, false otherwise.
auto intersects(std::pair<polydata *, vtkMatrix4x4 *> input0,
                std::pair<polydata *, vtkMatrix4x4 *> input1) -> bool;

} // namespace tf::vtk
