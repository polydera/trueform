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

/// @file chamfer_error.hpp
/// @brief One-way Chamfer error between point sets.
///
/// Computes the mean nearest-neighbor distance from source to target.
/// This is an asymmetric measure; for symmetric Chamfer distance,
/// compute both directions and average.

/// @brief Compute one-way Chamfer error from source to target.
/// @param source The source polydata.
/// @param target The target polydata.
/// @return Mean nearest-neighbor distance.
auto chamfer_error(polydata *source, polydata *target) -> float;

/// @brief Compute one-way Chamfer error from transformed source to target.
/// @param source The source polydata with transform.
/// @param target The target polydata.
/// @return Mean nearest-neighbor distance.
auto chamfer_error(std::pair<polydata *, vtkMatrix4x4 *> source,
                   polydata *target) -> float;

/// @brief Compute one-way Chamfer error from source to transformed target.
/// @param source The source polydata.
/// @param target The target polydata with transform.
/// @return Mean nearest-neighbor distance.
auto chamfer_error(polydata *source,
                   std::pair<polydata *, vtkMatrix4x4 *> target) -> float;

/// @brief Compute one-way Chamfer error between two transformed polydata.
/// @param source The source polydata with transform.
/// @param target The target polydata with transform.
/// @return Mean nearest-neighbor distance.
auto chamfer_error(std::pair<polydata *, vtkMatrix4x4 *> source,
                   std::pair<polydata *, vtkMatrix4x4 *> target) -> float;

} // namespace tf::vtk
