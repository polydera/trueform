/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/neighbor_result.hpp>
#include <optional>
#include <utility>

class vtkMatrix4x4;

namespace tf::vtk {

/// @file neighbor_search.hpp
/// @brief Nearest neighbor queries on polydata meshes.
///
/// Queries automatically select the appropriate primitive type based on
/// what the polydata contains, in this order:
/// 1. Polygons (if GetNumberOfPolys() > 0)
/// 2. Lines/segments (else if GetNumberOfLines() > 0)
/// 3. Points (otherwise)
///
/// Overloads without a radius parameter always return a result.
/// Overloads with a radius parameter return std::optional (may not find
/// anything within the specified radius).

// ============================================================================
// Form vs Point
// ============================================================================

/// @brief Find the nearest point on a mesh to a query point.
/// @param input The polydata mesh.
/// @param point The query point.
/// @return Neighbor result.
auto neighbor_search(polydata *input, tf::point<float, 3> point)
    -> neighbor_result;

/// @brief Find the nearest point on a mesh within a radius.
/// @param input The polydata mesh.
/// @param point The query point.
/// @param radius Maximum search radius.
/// @return Neighbor result if found within radius, nullopt otherwise.
auto neighbor_search(polydata *input, tf::point<float, 3> point, float radius)
    -> std::optional<neighbor_result>;

/// @brief Find the nearest point on a transformed mesh to a query point.
/// @param input The polydata mesh with transform.
/// @param point The query point.
/// @return Neighbor result.
auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input,
                     tf::point<float, 3> point) -> neighbor_result;

/// @brief Find the nearest point on a transformed mesh within a radius.
/// @param input The polydata mesh with transform.
/// @param point The query point.
/// @param radius Maximum search radius.
/// @return Neighbor result if found within radius, nullopt otherwise.
auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input,
                     tf::point<float, 3> point, float radius)
    -> std::optional<neighbor_result>;

// ============================================================================
// Form vs Form
// ============================================================================

/// @brief Find the closest pair of points between two meshes.
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh.
/// @return Neighbor pair result.
auto neighbor_search(polydata *input0, polydata *input1)
    -> neighbor_pair_result;

/// @brief Find the closest pair of points between two meshes within a radius.
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh.
/// @param radius Maximum search radius.
/// @return Neighbor pair result if found within radius, nullopt otherwise.
auto neighbor_search(polydata *input0, polydata *input1, float radius)
    -> std::optional<neighbor_pair_result>;

/// @brief Find the closest pair of points between two meshes (first transformed).
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh.
/// @return Neighbor pair result.
auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     polydata *input1) -> neighbor_pair_result;

/// @brief Find the closest pair of points between two meshes (first transformed) within a radius.
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh.
/// @param radius Maximum search radius.
/// @return Neighbor pair result if found within radius, nullopt otherwise.
auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     polydata *input1, float radius)
    -> std::optional<neighbor_pair_result>;

/// @brief Find the closest pair of points between two meshes (second transformed).
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh with transform.
/// @return Neighbor pair result.
auto neighbor_search(polydata *input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> neighbor_pair_result;

/// @brief Find the closest pair of points between two meshes (second transformed) within a radius.
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh with transform.
/// @param radius Maximum search radius.
/// @return Neighbor pair result if found within radius, nullopt otherwise.
auto neighbor_search(polydata *input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1, float radius)
    -> std::optional<neighbor_pair_result>;

/// @brief Find the closest pair of points between two meshes (both transformed).
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh with transform.
/// @return Neighbor pair result.
auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> neighbor_pair_result;

/// @brief Find the closest pair of points between two meshes (both transformed) within a radius.
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh with transform.
/// @param radius Maximum search radius.
/// @return Neighbor pair result if found within radius, nullopt otherwise.
auto neighbor_search(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1, float radius)
    -> std::optional<neighbor_pair_result>;

} // namespace tf::vtk
