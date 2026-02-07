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
#include <trueform/cut/return_curves.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <utility>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Embed intersection curves from mesh1 into mesh0.
///
/// Returns a polydata containing mesh0 with faces split along intersection
/// curves with mesh1. No faces from mesh1 are included.
///
/// @param input0 Mesh to embed curves into (must be tf::vtk::polydata).
/// @param input1 Mesh providing the cutting surface (must be tf::vtk::polydata).
/// @return Result polydata with embedded curves.
auto embedded_intersection_curves(polydata *input0, polydata *input1)
    -> vtkSmartPointer<polydata>;

/// @brief Embed intersection curves with transform on first mesh.
auto embedded_intersection_curves(std::pair<polydata *, vtkMatrix4x4 *> input0,
                                  polydata *input1)
    -> vtkSmartPointer<polydata>;

/// @brief Embed intersection curves with transform on second mesh.
auto embedded_intersection_curves(polydata *input0,
                                  std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> vtkSmartPointer<polydata>;

/// @brief Embed intersection curves with transforms on both meshes.
auto embedded_intersection_curves(std::pair<polydata *, vtkMatrix4x4 *> input0,
                                  std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> vtkSmartPointer<polydata>;

/// @brief Embed intersection curves and return the curves.
///
/// @param input0 Mesh to embed curves into (must be tf::vtk::polydata).
/// @param input1 Mesh providing the cutting surface (must be tf::vtk::polydata).
/// @param tag Pass tf::return_curves to get intersection curves.
/// @return Pair of (result polydata, curves polydata).
auto embedded_intersection_curves(polydata *input0, polydata *input1,
                                  tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

/// @brief Embed intersection curves with transform on first mesh and curves.
auto embedded_intersection_curves(std::pair<polydata *, vtkMatrix4x4 *> input0,
                                  polydata *input1, tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

/// @brief Embed intersection curves with transform on second mesh and curves.
auto embedded_intersection_curves(polydata *input0,
                                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                                  tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

/// @brief Embed intersection curves with transforms on both meshes and curves.
auto embedded_intersection_curves(std::pair<polydata *, vtkMatrix4x4 *> input0,
                                  std::pair<polydata *, vtkMatrix4x4 *> input1,
                                  tf::return_curves_t)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

} // namespace tf::vtk
