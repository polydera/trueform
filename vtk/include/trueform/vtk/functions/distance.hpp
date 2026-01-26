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
#include <trueform/vtk/functions/neighbor_search.hpp>

namespace tf::vtk {

/// @file distance.hpp
/// @brief Distance queries on polydata meshes.
///
/// Convenience wrappers around neighbor_search that return only the squared distance.
/// See neighbor_search.hpp for primitive type selection behavior.

// ============================================================================
// Form vs Point
// ============================================================================

/// @brief Compute squared distance from a mesh to a query point.
/// @param input The polydata mesh.
/// @param point The query point.
/// @return Squared distance to the nearest point on the mesh.
inline auto distance2(polydata *input, tf::point<float, 3> point) -> float {
  return neighbor_search(input, point).info.metric;
}

/// @brief Compute squared distance from a transformed mesh to a query point.
/// @param input The polydata mesh with transform.
/// @param point The query point.
/// @return Squared distance to the nearest point on the mesh.
inline auto distance2(std::pair<polydata *, vtkMatrix4x4 *> input,
                      tf::point<float, 3> point) -> float {
  return neighbor_search(input, point).info.metric;
}

// ============================================================================
// Form vs Form
// ============================================================================

/// @brief Compute squared distance between two meshes.
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh.
/// @return Squared distance between the closest points on the two meshes.
inline auto distance2(polydata *input0, polydata *input1) -> float {
  return neighbor_search(input0, input1).info.metric;
}

/// @brief Compute squared distance between two meshes (first transformed).
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh.
/// @return Squared distance between the closest points on the two meshes.
inline auto distance2(std::pair<polydata *, vtkMatrix4x4 *> input0,
                      polydata *input1) -> float {
  return neighbor_search(input0, input1).info.metric;
}

/// @brief Compute squared distance between two meshes (second transformed).
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh with transform.
/// @return Squared distance between the closest points on the two meshes.
inline auto distance2(polydata *input0,
                      std::pair<polydata *, vtkMatrix4x4 *> input1) -> float {
  return neighbor_search(input0, input1).info.metric;
}

/// @brief Compute squared distance between two meshes (both transformed).
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh with transform.
/// @return Squared distance between the closest points on the two meshes.
inline auto distance2(std::pair<polydata *, vtkMatrix4x4 *> input0,
                      std::pair<polydata *, vtkMatrix4x4 *> input1) -> float {
  return neighbor_search(input0, input1).info.metric;
}

// ============================================================================
// Distance (non-squared convenience wrappers)
// ============================================================================

/// @brief Compute distance from a mesh to a query point.
/// @param input The polydata mesh.
/// @param point The query point.
/// @return Distance to the nearest point on the mesh.
inline auto distance(polydata *input, tf::point<float, 3> point) -> float {
  return tf::sqrt(distance2(input, point));
}

/// @brief Compute distance from a transformed mesh to a query point.
/// @param input The polydata mesh with transform.
/// @param point The query point.
/// @return Distance to the nearest point on the mesh.
inline auto distance(std::pair<polydata *, vtkMatrix4x4 *> input,
                     tf::point<float, 3> point) -> float {
  return tf::sqrt(distance2(input, point));
}

/// @brief Compute distance between two meshes.
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh.
/// @return Distance between the closest points on the two meshes.
inline auto distance(polydata *input0, polydata *input1) -> float {
  return tf::sqrt(distance2(input0, input1));
}

/// @brief Compute distance between two meshes (first transformed).
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh.
/// @return Distance between the closest points on the two meshes.
inline auto distance(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     polydata *input1) -> float {
  return tf::sqrt(distance2(input0, input1));
}

/// @brief Compute distance between two meshes (second transformed).
/// @param input0 The first polydata mesh.
/// @param input1 The second polydata mesh with transform.
/// @return Distance between the closest points on the two meshes.
inline auto distance(polydata *input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1) -> float {
  return tf::sqrt(distance2(input0, input1));
}

/// @brief Compute distance between two meshes (both transformed).
/// @param input0 The first polydata mesh with transform.
/// @param input1 The second polydata mesh with transform.
/// @return Distance between the closest points on the two meshes.
inline auto distance(std::pair<polydata *, vtkMatrix4x4 *> input0,
                     std::pair<polydata *, vtkMatrix4x4 *> input1) -> float {
  return tf::sqrt(distance2(input0, input1));
}

} // namespace tf::vtk
