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
#include <trueform/vtk/functions/neighbor_result.hpp>
#include <utility>
#include <vector>

class vtkMatrix4x4;

namespace tf::vtk {

// ============================================================================
// kNN: Form vs Point (no radius)
// ============================================================================

/// @brief Find k nearest points on a mesh to a query point.
/// @param input The polydata mesh.
/// @param point The query point.
/// @param k Maximum number of neighbors to find.
/// @return Vector of neighbor results sorted by metric (squared distance). May contain fewer than k.
auto neighbor_search_k(polydata *input, tf::point<float, 3> point, std::size_t k)
    -> std::vector<neighbor_result>;

/// @brief Find k nearest points on a transformed mesh to a query point.
/// @param input The polydata mesh with transform.
/// @param point The query point.
/// @param k Maximum number of neighbors to find.
/// @return Vector of neighbor results sorted by metric (squared distance). May contain fewer than k.
auto neighbor_search_k(std::pair<polydata *, vtkMatrix4x4 *> input,
                       tf::point<float, 3> point, std::size_t k)
    -> std::vector<neighbor_result>;

// ============================================================================
// kNN: Form vs Point (with radius)
// ============================================================================

/// @brief Find k nearest points on a mesh within a radius.
/// @param input The polydata mesh.
/// @param point The query point.
/// @param k Maximum number of neighbors to find.
/// @param radius Maximum search radius.
/// @return Vector of neighbor results sorted by metric (squared distance). May contain fewer than k.
auto neighbor_search_k(polydata *input, tf::point<float, 3> point, std::size_t k,
                       float radius) -> std::vector<neighbor_result>;

/// @brief Find k nearest points on a transformed mesh within a radius.
/// @param input The polydata mesh with transform.
/// @param point The query point.
/// @param k Maximum number of neighbors to find.
/// @param radius Maximum search radius.
/// @return Vector of neighbor results sorted by metric (squared distance). May contain fewer than k.
auto neighbor_search_k(std::pair<polydata *, vtkMatrix4x4 *> input,
                       tf::point<float, 3> point, std::size_t k, float radius)
    -> std::vector<neighbor_result>;

} // namespace tf::vtk
