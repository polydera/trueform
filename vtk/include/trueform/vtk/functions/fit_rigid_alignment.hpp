/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/vtk/core/polydata.hpp>
#include <utility>
#include <vtkSmartPointer.h>

class vtkMatrix4x4;

namespace tf::vtk {

/// @file fit_rigid_alignment.hpp
/// @brief Rigid alignment between corresponding point sets.
///
/// Computes the optimal rigid transformation T such that T(source) ≈ target
/// using the Kabsch/Procrustes algorithm. Requires point correspondences
/// (same vertex count, same order).

/// @brief Fit rigid alignment from source to target.
/// @param source The source polydata.
/// @param target The target polydata (must have same vertex count).
/// @return Transformation matrix mapping source to target.
auto fit_rigid_alignment(polydata *source, polydata *target)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit rigid alignment from transformed source to target.
/// @param source The source polydata with transform.
/// @param target The target polydata.
/// @return Transformation matrix mapping source to target.
auto fit_rigid_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                         polydata *target) -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit rigid alignment from source to transformed target.
/// @param source The source polydata.
/// @param target The target polydata with transform.
/// @return Transformation matrix mapping source to target.
auto fit_rigid_alignment(polydata *source,
                         std::pair<polydata *, vtkMatrix4x4 *> target)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit rigid alignment between two transformed polydata.
/// @param source The source polydata with transform.
/// @param target The target polydata with transform.
/// @return Transformation matrix mapping source to target.
auto fit_rigid_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                         std::pair<polydata *, vtkMatrix4x4 *> target)
    -> vtkSmartPointer<vtkMatrix4x4>;

} // namespace tf::vtk
