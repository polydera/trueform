/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/vtk/functions/neighbor_search.hpp>

namespace tf::vtk {

/// @file distance.hpp
/// @brief Distance queries on polydata meshes.
///
/// Convenience wrappers around neighbor_search that return only the distance.
/// See neighbor_search.hpp for primitive type selection behavior.

// ============================================================================
// Form vs Point
// ============================================================================

/// @brief Compute distance from a mesh to a query point.
/// @param input The polydata mesh.
/// @param point The query point.
/// @return Distance to the nearest point on the mesh.
inline auto distance(polydata *input, tf::point<float, 3> point) -> float {
  return neighbor_search(input, point).distance;
}

/// @brief Compute distance from a transformed mesh to a query point.
/// @param input The polydata mesh with transform.
/// @param point The query point.
/// @return Distance to the nearest point on the mesh.
inline auto distance(std::pair<polydata *, vtkMatrix4x4 *> input,
                     tf::point<float, 3> point) -> float {
  return neighbor_search(input, point).distance;
}

// ============================================================================
// Form vs Form
// ============================================================================

/// @brief Compute distance between two meshes.
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh.
/// @return Distance between the closest points on the two meshes.
inline auto distance(polydata *input0, polydata *input1) -> float {
  return neighbor_search(input0, input1).distance;
}

/// @brief Compute distance between two meshes (first transformed).
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh.
/// @return Distance between the closest points on the two meshes.
inline auto distance(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     polydata *input1) -> float {
  return neighbor_search(input0, input1).distance;
}

/// @brief Compute distance between two meshes (second transformed).
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh with transform.
/// @return Distance between the closest points on the two meshes.
inline auto distance(polydata *input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1) -> float {
  return neighbor_search(input0, input1).distance;
}

/// @brief Compute distance between two meshes (both transformed).
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh with transform.
/// @return Distance between the closest points on the two meshes.
inline auto distance(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1) -> float {
  return neighbor_search(input0, input1).distance;
}

} // namespace tf::vtk
