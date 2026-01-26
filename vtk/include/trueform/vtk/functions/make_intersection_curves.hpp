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
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Compute intersection curves between two meshes.
///
/// Returns a polydata containing line cells representing the curves
/// where the two input meshes intersect.
///
/// @param input0 First input mesh (must be tf::vtk::polydata).
/// @param input1 Second input mesh (must be tf::vtk::polydata).
/// @return Polydata with intersection curves as lines, or nullptr on failure.
///
/// @code
/// auto curves = tf::vtk::make_intersection_curves(mesh0, mesh1);
/// @endcode
auto make_intersection_curves(polydata *input0, polydata *input1)
    -> vtkSmartPointer<polydata>;

/// @brief Compute intersection curves with transform on first mesh.
///
/// @param input0 Pair of (mesh, transform) for first input.
/// @param input1 Second input mesh.
/// @return Polydata with intersection curves as lines, or nullptr on failure.
auto make_intersection_curves(std::pair<polydata *, vtkMatrix4x4 *> input0,
                              polydata *input1)
    -> vtkSmartPointer<polydata>;

/// @brief Compute intersection curves with transform on second mesh.
///
/// @param input0 First input mesh.
/// @param input1 Pair of (mesh, transform) for second input.
/// @return Polydata with intersection curves as lines, or nullptr on failure.
auto make_intersection_curves(polydata *input0,
                              std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> vtkSmartPointer<polydata>;

/// @brief Compute intersection curves with transforms on both meshes.
///
/// @param input0 Pair of (mesh, transform) for first input.
/// @param input1 Pair of (mesh, transform) for second input.
/// @return Polydata with intersection curves as lines, or nullptr on failure.
///
/// @code
/// auto curves = tf::vtk::make_intersection_curves(
///     {mesh0, transform0}, {mesh1, transform1});
/// @endcode
auto make_intersection_curves(std::pair<polydata *, vtkMatrix4x4 *> input0,
                              std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> vtkSmartPointer<polydata>;

} // namespace tf::vtk
