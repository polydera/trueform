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

#include "../core/buffer.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/points_buffer.hpp"
#include "../core/policy/normals.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "./impl/fit_rigid_alignment_point_to_plane.hpp"

#include <type_traits>

namespace tf::geometry {

/// @brief Workspace for point-to-point kNN alignment.
///
/// Uses indices into source to preserve all source policies (including normals)
/// through correspondence sorting and filtering.
template <typename T, std::size_t Dims> struct knn_alignment_point_state {
  tf::buffer<int> src_indices;
  tf::points_buffer<T, Dims> target_points;
  tf::buffer<T> distances;
};

/// @brief Workspace for point-to-plane kNN alignment.
///
/// Uses indices into source to preserve all source policies (including normals)
/// through correspondence sorting and filtering. When source has normals,
/// they are automatically used for weighting in the fitting step.
template <typename T, std::size_t Dims> struct knn_alignment_plane_state {
  tf::buffer<int> src_indices;
  tf::points_buffer<T, Dims> target_points;
  tf::unit_vectors_buffer<T, Dims> target_normals;
  tf::buffer<T> distances;
  plane_alignment_state<T> alignment_state;
};
} // namespace tf::geometry
namespace tf {

/// @brief kNN alignment state, automatically selected based on policies.
///
/// Inherits from the appropriate state type based on whether the target has
/// normals:
/// - Target has no normals → point-to-point
/// - Target has normals → point-to-plane (with automatic source normal
///   weighting if source also has normals)
template <typename Policy0, typename Policy1>
using knn_alignment_state = std::conditional_t<
    !tf::has_normals_policy<Policy1>,
    geometry::knn_alignment_point_state<tf::coordinate_type<Policy0, Policy1>,
                                        tf::coordinate_dims_v<Policy0>>,
    geometry::knn_alignment_plane_state<tf::coordinate_type<Policy0, Policy1>,
                                        tf::coordinate_dims_v<Policy0>>>;

/// @brief Factory function to create the appropriate kNN alignment state.
///
/// Returns the correct state type based on the policies of the input point
/// sets. Use this to avoid manually specifying template parameters.
///
/// @param X Source point set (policies inspected but not stored).
/// @param Y Target point set (policies inspected but not stored).
/// @return Appropriate state struct for the given point set policies.
template <typename Policy0, typename Policy1>
auto make_knn_alignment_state(const tf::points<Policy0> & /*X*/,
                              const tf::points<Policy1> & /*Y*/) {
  return knn_alignment_state<Policy0, Policy1>{};
}

} // namespace tf
