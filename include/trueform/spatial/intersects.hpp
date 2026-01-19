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

#include "../core/intersects.hpp"
#include "./policy/tree.hpp"
#include "./search.hpp"

namespace tf {

/// @ingroup spatial_queries
/// @brief Test whether two forms intersect.
///
/// Uses dual-tree traversal to find any intersecting pair of primitives.
///
/// @param form0 The first form.
/// @param form1 The second form.
/// @return True if any primitives intersect.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "First form must have a tree policy attached. Use: form | tf::tag(tree)");
  static_assert(tf::has_tree_policy<Policy1>,
                "Second form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::search(form0, form1, tf::intersects_f, tf::intersects_f);
}

/// @ingroup spatial_queries
/// @brief Test whether a form contains a point.
///
/// @param form The spatial form to query.
/// @param obj The query point.
/// @return True if any primitive contains the point.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::point_like<Dims, Policy1> &obj) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

/// @ingroup spatial_queries
/// @brief Test whether a form intersects a polygon.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::polygon<Dims, Policy1> &obj) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto obj_aabb = tf::aabb_from(obj);
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj_aabb); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

/// @ingroup spatial_queries
/// @brief Test whether a form intersects a segment.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::segment<Dims, Policy1> &obj) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

/// @ingroup spatial_queries
/// @brief Test whether a form intersects a ray.
template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::ray_like<Dims, Policy> &obj) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

/// @ingroup spatial_queries
/// @brief Test whether a form intersects a line.
template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::line_like<Dims, Policy> &obj) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

/// @ingroup spatial_queries
/// @brief Test whether a form intersects a plane.
template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::plane_like<Dims, Policy> &obj) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

/// @ingroup spatial_queries
/// @brief Test whether a form intersects an AABB.
template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::aabb_like<Dims, Policy> &obj) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

/// @ingroup spatial_queries
/// @brief Test whether a point intersects a form (symmetric).
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::point_like<Dims, Policy0> &obj,
                const tf::form<Dims, Policy1> &form) -> bool {
  return tf::intersects(form, obj);
}

/// @ingroup spatial_queries
/// @brief Test whether a polygon intersects a form (symmetric).
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &obj,
                const tf::form<Dims, Policy1> &form) -> bool {
  return tf::intersects(form, obj);
}

/// @ingroup spatial_queries
/// @brief Test whether a segment intersects a form (symmetric).
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::segment<Dims, Policy0> &obj,
                const tf::form<Dims, Policy1> &form) -> bool {
  return tf::intersects(form, obj);
}

/// @ingroup spatial_queries
/// @brief Test whether a ray intersects a form (symmetric).
template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::ray_like<Dims, Policy0> &obj,
                const tf::form<Dims, Policy> &form) -> bool {
  return tf::intersects(form, obj);
}

/// @ingroup spatial_queries
/// @brief Test whether a line intersects a form (symmetric).
template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::line_like<Dims, Policy0> &obj,
                const tf::form<Dims, Policy> &form) -> bool {
  return tf::intersects(form, obj);
}

/// @ingroup spatial_queries
/// @brief Test whether a plane intersects a form (symmetric).
template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::plane_like<Dims, Policy0> &obj,
                const tf::form<Dims, Policy> &form) -> bool {
  return tf::intersects(form, obj);
}

/// @ingroup spatial_queries
/// @brief Test whether an AABB intersects a form (symmetric).
template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::aabb_like<Dims, Policy0> &obj,
                const tf::form<Dims, Policy> &form) -> bool {
  return tf::intersects(form, obj);
}

} // namespace tf
