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

#include "../core/aabb_from.hpp"
#include "../core/closest_metric_point.hpp"
#include "../core/closest_metric_point_pair.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/line_like.hpp"
#include "../core/polygon.hpp"
#include "../core/ray_like.hpp"
#include "../core/segment.hpp"
#include "./policy/tree.hpp"
#include "./tree/traversal_metric.hpp"
#include "./tree_search/nearness_search.hpp"

namespace tf {

// ============================================================================
// Point overloads
// ============================================================================

/// @ingroup spatial_queries
/// @brief Find the nearest neighbor to a point in a form.
///
/// Traverses the spatial tree to find the primitive closest to the query point.
///
/// @param form The spatial form to query.
/// @param obj The query point.
/// @return A @ref tf::tree_metric_info containing the nearest primitive ID,
///         squared distance, and closest point on the primitive.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

/// @ingroup spatial_queries
/// @brief Find the nearest neighbor within a radius.
///
/// @param form The spatial form to query.
/// @param obj The query point.
/// @param radius Maximum search radius (squared distance).
/// @return Result containing the nearest primitive, or invalid if none within radius.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

/// @ingroup spatial_queries
/// @brief Find k nearest neighbors to a point.
///
/// @param form The spatial form to query.
/// @param obj The query point.
/// @param knn A @ref tf::nearest_neighbors buffer to store results.
/// @return The updated knn buffer containing the k nearest primitives.
template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Segment overloads (use aabb_from for BV metric)
// ============================================================================

/// @ingroup spatial_queries
/// @brief Find the nearest neighbor to a segment.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Ray overloads
// ============================================================================

/// @ingroup spatial_queries
/// @brief Find the nearest neighbor to a ray.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Line overloads
// ============================================================================

/// @ingroup spatial_queries
/// @brief Find the nearest neighbor to a line.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Plane overloads
// ============================================================================

/// @ingroup spatial_queries
/// @brief Find the nearest neighbor to a plane.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::plane_like<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::plane_like<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::plane_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::plane_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Polygon overloads (3D uses tag_plane, 2D uses obj directly)
// ============================================================================

/// @ingroup spatial_queries
/// @brief Find the nearest neighbor to a polygon.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  auto plane_obj = tf::tag_plane(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, plane_obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  auto plane_obj = tf::tag_plane(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, plane_obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  auto plane_obj = tf::tag_plane(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, plane_obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// 2D polygon overloads (use obj directly, not tag_plane)
template <typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<2, Policy0> &form,
                     const tf::polygon<2, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<2, Policy0> &form,
                     const tf::polygon<2, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <typename Policy0, typename Policy1, typename RandomIt>
auto neighbor_search(const tf::form<2, Policy0> &form,
                     const tf::polygon<2, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <typename Policy0, typename Policy1, typename RandomIt>
auto neighbor_search(const tf::form<2, Policy0> &form,
                     const tf::polygon<2, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Form-to-form overloads (dual tree)
// ============================================================================

/// @ingroup spatial_queries
/// @brief Find the nearest pair of primitives between two forms.
///
/// Uses dual-tree traversal to efficiently find the closest pair.
///
/// @param form0 The first form.
/// @param form1 The second form.
/// @return A @ref tf::tree_metric_info_pair containing IDs from both forms,
///         squared distance, and closest points on each primitive.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1) {
  static_assert(tf::has_tree_policy<Policy0>,
                "First form must have a tree policy attached. Use: form | tf::tag(tree)");
  static_assert(tf::has_tree_policy<Policy1>,
                "Second form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form0, form1, [](const auto &obj0, const auto &obj1) {
        return tf::closest_metric_point_pair(obj0, obj1);
      });
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  static_assert(tf::has_tree_policy<Policy0>,
                "First form must have a tree policy attached. Use: form | tf::tag(tree)");
  static_assert(tf::has_tree_policy<Policy1>,
                "Second form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::nearness_search(
      form0, form1,
      [](const auto &obj0, const auto &obj1) {
        return tf::closest_metric_point_pair(obj0, obj1);
      },
      radius);
}

} // namespace tf
