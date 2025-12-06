/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/core.hpp>
#include <vtkSmartPointer.h>

class vtkTransform;

namespace tf::vtk {

/// @brief Creates vtkTransform from a trueform transformation (float).
/// @param transform Trueform transformation (3x4 efficient format).
/// @return A new vtkTransform object.
auto make_vtk_transform(const tf::transformation<float, 3> &transform)
    -> vtkSmartPointer<vtkTransform>;

/// @brief Creates vtkTransform from a trueform transformation (double).
/// @param transform Trueform transformation (3x4 efficient format).
/// @return A new vtkTransform object.
auto make_vtk_transform(const tf::transformation<double, 3> &transform)
    -> vtkSmartPointer<vtkTransform>;

/// @brief Fills existing vtkTransform from a trueform transformation (float).
/// @param transform Target vtkTransform to fill.
/// @param tf_transform Trueform transformation (3x4 efficient format).
auto fill_vtk_transform(vtkTransform *transform,
                        const tf::transformation<float, 3> &tf_transform) -> void;

/// @brief Fills existing vtkTransform from a trueform transformation (double).
/// @param transform Target vtkTransform to fill.
/// @param tf_transform Trueform transformation (3x4 efficient format).
auto fill_vtk_transform(vtkTransform *transform,
                        const tf::transformation<double, 3> &tf_transform) -> void;

} // namespace tf::vtk
