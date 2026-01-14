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

/// @file fit_knn_alignment.hpp
/// @brief k-NN based alignment (ICP step).
///
/// For each point in source, finds the k nearest neighbors in target
/// and computes a weighted correspondence. This is one iteration of ICP
/// when k=1. For k>1, soft correspondences provide robustness to noise.

/// @brief Fit k-NN alignment from source to target.
/// @param source The source polydata.
/// @param target The target polydata.
/// @param k Number of nearest neighbors (default: 1 = classic ICP).
/// @param sigma Gaussian kernel width. If negative, uses adaptive scaling.
/// @return Transformation matrix mapping source to target.
auto fit_knn_alignment(polydata *source, polydata *target, std::size_t k = 1,
                       float sigma = -1) -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit k-NN alignment from transformed source to target.
/// @param source The source polydata with transform.
/// @param target The target polydata.
/// @param k Number of nearest neighbors.
/// @param sigma Gaussian kernel width.
/// @return Transformation matrix mapping source to target.
auto fit_knn_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                       polydata *target, std::size_t k = 1, float sigma = -1)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit k-NN alignment from source to transformed target.
/// @param source The source polydata.
/// @param target The target polydata with transform.
/// @param k Number of nearest neighbors.
/// @param sigma Gaussian kernel width.
/// @return Transformation matrix mapping source to target.
auto fit_knn_alignment(polydata *source,
                       std::pair<polydata *, vtkMatrix4x4 *> target,
                       std::size_t k = 1, float sigma = -1)
    -> vtkSmartPointer<vtkMatrix4x4>;

/// @brief Fit k-NN alignment between two transformed polydata.
/// @param source The source polydata with transform.
/// @param target The target polydata with transform.
/// @param k Number of nearest neighbors.
/// @param sigma Gaussian kernel width.
/// @return Transformation matrix mapping source to target.
auto fit_knn_alignment(std::pair<polydata *, vtkMatrix4x4 *> source,
                       std::pair<polydata *, vtkMatrix4x4 *> target,
                       std::size_t k = 1, float sigma = -1)
    -> vtkSmartPointer<vtkMatrix4x4>;

} // namespace tf::vtk
