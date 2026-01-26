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
#include <trueform/cut/boolean_op.hpp>
#include <trueform/cut/return_curves.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <utility>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Compute boolean operation between two meshes.
///
/// Returns a polydata with "Labels" cell scalars indicating face origin
/// (positive for mesh0, negative for mesh1).
///
/// @param input0 First input mesh (must be tf::vtk::polydata).
/// @param input1 Second input mesh (must be tf::vtk::polydata).
/// @param op Boolean operation to perform.
/// @return Result polydata with Labels cell data.
auto make_boolean(polydata *input0, polydata *input1, tf::boolean_op op)
    -> vtkSmartPointer<polydata>;

/// @brief Compute boolean operation with transform on first mesh.
auto make_boolean(std::pair<polydata *, vtkMatrix4x4 *> input0,
                  polydata *input1, tf::boolean_op op)
    -> vtkSmartPointer<polydata>;

/// @brief Compute boolean operation with transform on second mesh.
auto make_boolean(polydata *input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op)
    -> vtkSmartPointer<polydata>;

/// @brief Compute boolean operation with transforms on both meshes.
auto make_boolean(std::pair<polydata *, vtkMatrix4x4 *> input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op)
    -> vtkSmartPointer<polydata>;

/// @brief Compute boolean operation with intersection curves.
///
/// Returns the result mesh and the intersection curves as separate polydata.
///
/// @param input0 First input mesh (must be tf::vtk::polydata).
/// @param input1 Second input mesh (must be tf::vtk::polydata).
/// @param op Boolean operation to perform.
/// @param tag Pass tf::return_curves to get intersection curves.
/// @return Pair of (result polydata with Labels cell data, curves polydata).
auto make_boolean(polydata *input0, polydata *input1, tf::boolean_op op,
                  tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

/// @brief Compute boolean operation with transform on first mesh and curves.
auto make_boolean(std::pair<polydata *, vtkMatrix4x4 *> input0,
                  polydata *input1, tf::boolean_op op, tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

/// @brief Compute boolean operation with transform on second mesh and curves.
auto make_boolean(polydata *input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op, tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

/// @brief Compute boolean operation with transforms on both meshes and curves.
auto make_boolean(std::pair<polydata *, vtkMatrix4x4 *> input0,
                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                  tf::boolean_op op, tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

} // namespace tf::vtk
