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
#include <trueform/core/offset_block_vector.hpp>
#include <trueform/vtk/core/make_points.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/neighbor_result.hpp>
#include <utility>

class vtkMatrix4x4;
class vtkPoints;

namespace tf::vtk {

// ============================================================================
// Batch kNN: Form vs Points (no radius)
// ============================================================================

/// @brief Find k nearest points on a mesh for multiple query points.
/// @param input The polydata mesh.
/// @param points Query points as vtkPoints.
/// @param k Maximum number of neighbors per query point.
/// @return Offset block vector of neighbor results (variable count per query).
auto neighbor_search_k_batch(polydata *input, vtkPoints *points, std::size_t k)
    -> tf::offset_block_vector<std::size_t, neighbor_result>;

/// @brief Find k nearest points on a mesh for multiple query points.
/// @param input The polydata mesh.
/// @param points Query points as tf::vtk::points_t.
/// @param k Maximum number of neighbors per query point.
/// @return Offset block vector of neighbor results (variable count per query).
auto neighbor_search_k_batch(polydata *input, points_t points, std::size_t k)
    -> tf::offset_block_vector<std::size_t, neighbor_result>;

/// @brief Find k nearest points on a transformed mesh for multiple query points.
/// @param input The polydata mesh with transform.
/// @param points Query points as vtkPoints.
/// @param k Maximum number of neighbors per query point.
/// @return Offset block vector of neighbor results (variable count per query).
auto neighbor_search_k_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                             vtkPoints *points, std::size_t k)
    -> tf::offset_block_vector<std::size_t, neighbor_result>;

/// @brief Find k nearest points on a transformed mesh for multiple query points.
/// @param input The polydata mesh with transform.
/// @param points Query points as tf::vtk::points_t.
/// @param k Maximum number of neighbors per query point.
/// @return Offset block vector of neighbor results (variable count per query).
auto neighbor_search_k_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                             points_t points, std::size_t k)
    -> tf::offset_block_vector<std::size_t, neighbor_result>;

// ============================================================================
// Batch kNN: Form vs Points (with radius)
// ============================================================================

/// @brief Find k nearest points on a mesh within a radius for multiple query points.
/// @param input The polydata mesh.
/// @param points Query points as vtkPoints.
/// @param k Maximum number of neighbors per query point.
/// @param radius Maximum search radius.
/// @return Offset block vector of neighbor results (variable count per query).
auto neighbor_search_k_batch(polydata *input, vtkPoints *points, std::size_t k,
                             float radius)
    -> tf::offset_block_vector<std::size_t, neighbor_result>;

/// @brief Find k nearest points on a mesh within a radius for multiple query points.
/// @param input The polydata mesh.
/// @param points Query points as tf::vtk::points_t.
/// @param k Maximum number of neighbors per query point.
/// @param radius Maximum search radius.
/// @return Offset block vector of neighbor results (variable count per query).
auto neighbor_search_k_batch(polydata *input, points_t points, std::size_t k,
                             float radius)
    -> tf::offset_block_vector<std::size_t, neighbor_result>;

/// @brief Find k nearest points on a transformed mesh within a radius for multiple query points.
/// @param input The polydata mesh with transform.
/// @param points Query points as vtkPoints.
/// @param k Maximum number of neighbors per query point.
/// @param radius Maximum search radius.
/// @return Offset block vector of neighbor results (variable count per query).
auto neighbor_search_k_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                             vtkPoints *points, std::size_t k, float radius)
    -> tf::offset_block_vector<std::size_t, neighbor_result>;

/// @brief Find k nearest points on a transformed mesh within a radius for multiple query points.
/// @param input The polydata mesh with transform.
/// @param points Query points as tf::vtk::points_t.
/// @param k Maximum number of neighbors per query point.
/// @param radius Maximum search radius.
/// @return Offset block vector of neighbor results (variable count per query).
auto neighbor_search_k_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                             points_t points, std::size_t k, float radius)
    -> tf::offset_block_vector<std::size_t, neighbor_result>;

} // namespace tf::vtk
