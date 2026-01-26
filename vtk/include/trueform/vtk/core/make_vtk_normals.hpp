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

class vtkFloatArray;

namespace tf::vtk {

/// @brief Creates vtkFloatArray normals from a unit_vectors_buffer (copies data).
/// @param normals Trueform unit vectors buffer.
/// @return A new vtkFloatArray with copied normals data.
auto make_vtk_normals(const tf::unit_vectors_buffer<float, 3> &normals)
    -> vtkSmartPointer<vtkFloatArray>;

/// @brief Creates vtkFloatArray normals from a unit_vectors_buffer (moves data).
/// @param normals Trueform unit vectors buffer (consumed).
/// @return A new vtkFloatArray with transferred ownership.
auto make_vtk_normals(tf::unit_vectors_buffer<float, 3> &&normals)
    -> vtkSmartPointer<vtkFloatArray>;

} // namespace tf::vtk
