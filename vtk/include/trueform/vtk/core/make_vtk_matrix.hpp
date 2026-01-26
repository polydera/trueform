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

class vtkMatrix4x4;

namespace tf::vtk {

/// @brief Creates vtkMatrix4x4 from a trueform transformation (float).
/// @param transform Trueform transformation (3x4 efficient format).
/// @return A new vtkMatrix4x4 object.
auto make_vtk_matrix(const tf::transformation<float, 3> &transform)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Creates vtkMatrix4x4 from a trueform transformation (double).
/// @param transform Trueform transformation (3x4 efficient format).
/// @return A new vtkMatrix4x4 object.
auto make_vtk_matrix(const tf::transformation<double, 3> &transform)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fills existing vtkMatrix4x4 from a trueform transformation (float).
/// @param matrix Target vtkMatrix4x4 to fill.
/// @param transform Trueform transformation (3x4 efficient format).
auto fill_vtk_matrix(vtkMatrix4x4 *matrix,
                     const tf::transformation<float, 3> &transform) -> void;

/// @brief Fills existing vtkMatrix4x4 from a trueform transformation (double).
/// @param matrix Target vtkMatrix4x4 to fill.
/// @param transform Trueform transformation (3x4 efficient format).
auto fill_vtk_matrix(vtkMatrix4x4 *matrix,
                     const tf::transformation<double, 3> &transform) -> void;

} // namespace tf::vtk
