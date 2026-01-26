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
#include <cstddef>
#include <utility>
#include <vtkSmartPointer.h>

class vtkMatrix4x4;

namespace tf::vtk {

/// @file fit_obb_alignment.hpp
/// @brief OBB-based alignment between point sets.
///
/// Computes a rigid alignment by matching oriented bounding boxes.
/// No point correspondences are needed. The 180° ambiguity is resolved
/// by testing all orientations and selecting the one with lowest
/// chamfer distance to the target.

/// @brief Fit OBB alignment from source to target.
/// @param source The source polydata.
/// @param target The target polydata.
/// @param sample_size Number of points to sample for disambiguation (default: 100).
/// @return Transformation matrix mapping source to target.
auto fit_obb_alignment(polydata *source, polydata *target,
                       std::size_t sample_size = 100)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit OBB alignment from transformed source to target.
/// @param source The source polydata with transform.
/// @param target The target polydata.
/// @param sample_size Number of points to sample for disambiguation.
/// @return Transformation matrix mapping source to target.
auto fit_obb_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                       polydata *target, std::size_t sample_size = 100)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit OBB alignment from source to transformed target.
/// @param source The source polydata.
/// @param target The target polydata with transform.
/// @param sample_size Number of points to sample for disambiguation.
/// @return Transformation matrix mapping source to target.
auto fit_obb_alignment(polydata *source,
                       std::pair<polydata *, vtkMatrix4x4 *> target,
                       std::size_t sample_size = 100)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit OBB alignment between two transformed polydata.
/// @param source The source polydata with transform.
/// @param target The target polydata with transform.
/// @param sample_size Number of points to sample for disambiguation.
/// @return Transformation matrix mapping source to target.
auto fit_obb_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                       std::pair<polydata *, vtkMatrix4x4 *> target,
                       std::size_t sample_size = 100)
    -> vtkSmartPointer<vtkMatrix4x4>;

} // namespace tf::vtk
