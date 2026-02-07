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
#include <trueform/geometry/icp_config.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <utility>
#include <vtkSmartPointer.h>

class vtkMatrix4x4;

namespace tf::vtk {

/// @file fit_icp_alignment.hpp
/// @brief Iterative Closest Point (ICP) alignment.
///
/// Iteratively refines a rigid transformation aligning source to target.
/// Handles subsampling, convergence detection, and outlier rejection.
///
/// If target has point normals, uses point-to-plane ICP (faster convergence).
/// If both source and target have normals, uses normal weighting.
///
/// Returns a DELTA transformation mapping source world coordinates to target
/// world coordinates. To get the total transformation for source local coords:
/// @code
/// auto delta = tf::vtk::fit_icp_alignment({source, T_initial}, target);
/// auto total = vtkMatrix4x4::New();
/// vtkMatrix4x4::Multiply4x4(delta, T_initial, total);
/// @endcode

/// @brief Fit ICP alignment from source to target.
/// @param source The source polydata.
/// @param target The target polydata.
/// @param config ICP configuration.
/// @return Delta transformation mapping source_world -> target_world.
auto fit_icp_alignment(polydata *source, polydata *target,
                       const tf::icp_config &config = {})
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit ICP alignment from transformed source to target.
/// @param source The source polydata with initial transform.
/// @param target The target polydata.
/// @param config ICP configuration.
/// @return Delta transformation mapping source_world -> target_world.
auto fit_icp_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                       polydata *target, const tf::icp_config &config = {})
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit ICP alignment from source to transformed target.
/// @param source The source polydata.
/// @param target The target polydata with transform.
/// @param config ICP configuration.
/// @return Delta transformation mapping source_world -> target_world.
auto fit_icp_alignment(polydata *source,
                       std::pair<polydata *, vtkMatrix4x4 *> target,
                       const tf::icp_config &config = {})
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit ICP alignment between two transformed polydata.
/// @param source The source polydata with initial transform.
/// @param target The target polydata with transform.
/// @param config ICP configuration.
/// @return Delta transformation mapping source_world -> target_world.
auto fit_icp_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                       std::pair<polydata *, vtkMatrix4x4 *> target,
                       const tf::icp_config &config = {})
    -> vtkSmartPointer<vtkMatrix4x4>;

} // namespace tf::vtk
