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
#include <trueform/vtk/core/make_points.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/neighbor_result.hpp>
#include <utility>
#include <vector>

class vtkMatrix4x4;
class vtkPoints;

namespace tf::vtk {

// ============================================================================
// Batch: Form vs Points (no radius)
// ============================================================================

/// @brief Find nearest points on a mesh for multiple query points.
/// @param input The polydata mesh.
/// @param points Query points as vtkPoints.
/// @return Vector of neighbor results. Each result is convertible to bool.
auto neighbor_search_batch(polydata *input, vtkPoints *points)
    -> std::vector<neighbor_result>;

/// @brief Find nearest points on a mesh for multiple query points.
/// @param input The polydata mesh.
/// @param points Query points as tf::vtk::points_t.
/// @return Vector of neighbor results. Each result is convertible to bool.
auto neighbor_search_batch(polydata *input, points_t points)
    -> std::vector<neighbor_result>;

/// @brief Find nearest points on a transformed mesh for multiple query points.
/// @param input The polydata mesh with transform.
/// @param points Query points as vtkPoints.
/// @return Vector of neighbor results. Each result is convertible to bool.
auto neighbor_search_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                           vtkPoints *points) -> std::vector<neighbor_result>;

/// @brief Find nearest points on a transformed mesh for multiple query points.
/// @param input The polydata mesh with transform.
/// @param points Query points as tf::vtk::points_t.
/// @return Vector of neighbor results. Each result is convertible to bool.
auto neighbor_search_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                           points_t points) -> std::vector<neighbor_result>;

// ============================================================================
// Batch: Form vs Points (with radius)
// ============================================================================

/// @brief Find nearest points on a mesh within a radius for multiple query points.
/// @param input The polydata mesh.
/// @param points Query points as vtkPoints.
/// @param radius Maximum search radius.
/// @return Vector of neighbor results. Each result is convertible to bool (false if not found within radius).
auto neighbor_search_batch(polydata *input, vtkPoints *points, float radius)
    -> std::vector<neighbor_result>;

/// @brief Find nearest points on a mesh within a radius for multiple query points.
/// @param input The polydata mesh.
/// @param points Query points as tf::vtk::points_t.
/// @param radius Maximum search radius.
/// @return Vector of neighbor results. Each result is convertible to bool (false if not found within radius).
auto neighbor_search_batch(polydata *input, points_t points, float radius)
    -> std::vector<neighbor_result>;

/// @brief Find nearest points on a transformed mesh within a radius for multiple query points.
/// @param input The polydata mesh with transform.
/// @param points Query points as vtkPoints.
/// @param radius Maximum search radius.
/// @return Vector of neighbor results. Each result is convertible to bool (false if not found within radius).
auto neighbor_search_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                           vtkPoints *points, float radius)
    -> std::vector<neighbor_result>;

/// @brief Find nearest points on a transformed mesh within a radius for multiple query points.
/// @param input The polydata mesh with transform.
/// @param points Query points as tf::vtk::points_t.
/// @param radius Maximum search radius.
/// @return Vector of neighbor results. Each result is convertible to bool (false if not found within radius).
auto neighbor_search_batch(std::pair<polydata *, vtkMatrix4x4 *> input,
                           points_t points, float radius)
    -> std::vector<neighbor_result>;

} // namespace tf::vtk
