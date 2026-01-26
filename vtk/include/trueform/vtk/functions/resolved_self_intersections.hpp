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
#include <tuple>

namespace tf::vtk {

/// @brief Resolve self-intersections in a mesh by embedding intersection curves.
///
/// Finds where the mesh intersects itself and splits faces along those curves,
/// creating a new mesh where self-intersection curves become edges.
///
/// @param input The polydata mesh.
/// @return A new polydata with self-intersections resolved.
auto resolved_self_intersections(polydata *input) -> vtkSmartPointer<polydata>;

/// @brief Resolve self-intersections and return the intersection curves.
///
/// @param input The polydata mesh.
/// @param tag Pass tf::return_curves to get the curves.
/// @return Tuple of (resolved mesh, intersection curves as polydata).
auto resolved_self_intersections(polydata *input, tf::return_curves_t)
    -> std::tuple<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>>;

} // namespace tf::vtk
