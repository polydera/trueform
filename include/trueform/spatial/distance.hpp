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

#include "./neighbor_search.hpp"
#include "./policy/tree.hpp"

namespace tf {

/// @ingroup spatial_queries
/// @brief Compute the squared distance from a form to a point.
///
/// Uses @ref tf::neighbor_search to find the closest primitive in the form.
///
/// @param form The spatial form to query.
/// @param obj The query point.
/// @return Squared distance to the nearest primitive.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::point_like<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @copydoc distance2(const tf::form<Dims, Policy0>&, const tf::point_like<Dims, Policy1>&)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::point_like<Dims, Policy0> &obj,
               const tf::form<Dims, Policy1> &form) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @ingroup spatial_queries
/// @brief Compute the Euclidean distance from a form to a point.
///
/// @param form The spatial form to query.
/// @param obj The query point.
/// @return Distance to the nearest primitive.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::point_like<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

/// @copydoc distance(const tf::form<Dims, Policy0>&, const tf::point_like<Dims, Policy1>&)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::point_like<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

/// @ingroup spatial_queries
/// @brief Compute the squared distance from a form to a segment.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::segment<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::segment<Dims, Policy0> &obj,
               const tf::form<Dims, Policy1> &form) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @ingroup spatial_queries
/// @brief Compute the Euclidean distance from a form to a segment.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::segment<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::segment<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

/// @ingroup spatial_queries
/// @brief Compute the squared distance from a form to a line.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::line_like<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::line_like<Dims, Policy0> &obj,
               const tf::form<Dims, Policy1> &form) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @ingroup spatial_queries
/// @brief Compute the Euclidean distance from a form to a line.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::line_like<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::line_like<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

/// @ingroup spatial_queries
/// @brief Compute the squared distance from a form to a ray.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::ray_like<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::ray_like<Dims, Policy0> &obj,
               const tf::form<Dims, Policy1> &form) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @ingroup spatial_queries
/// @brief Compute the Euclidean distance from a form to a ray.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::ray_like<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::ray_like<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

/// @ingroup spatial_queries
/// @brief Compute the squared distance from a form to a polygon.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::polygon<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::polygon<Dims, Policy0> &obj,
               const tf::form<Dims, Policy1> &form) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @ingroup spatial_queries
/// @brief Compute the Euclidean distance from a form to a polygon.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::polygon<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::polygon<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

/// @ingroup spatial_queries
/// @brief Compute the squared distance from a form to a plane.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form,
               const tf::plane_like<Dims, Policy1> &obj) {
  static_assert(tf::has_tree_policy<Policy0>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::plane_like<Dims, Policy0> &obj,
               const tf::form<Dims, Policy1> &form) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form, obj).metric();
}

/// @ingroup spatial_queries
/// @brief Compute the Euclidean distance from a form to a plane.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form,
              const tf::plane_like<Dims, Policy1> &obj) {
  return tf::sqrt(distance2(form, obj));
}

/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::plane_like<Dims, Policy0> &obj,
              const tf::form<Dims, Policy0> &form) {
  return tf::sqrt(distance2(form, obj));
}

/// @ingroup spatial_queries
/// @brief Compute the squared distance between two forms.
///
/// Uses dual-tree traversal to find the closest pair of primitives.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::form<Dims, Policy0> &form0,
               const tf::form<Dims, Policy1> &form1) {
  static_assert(tf::has_tree_policy<Policy0>,
                "First form must have a tree policy attached. Use: form | tf::tag(tree)");
  static_assert(tf::has_tree_policy<Policy1>,
                "Second form must have a tree policy attached. Use: form | tf::tag(tree)");
  return neighbor_search(form0, form1).metric();
}

/// @ingroup spatial_queries
/// @brief Compute the Euclidean distance between two forms.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::form<Dims, Policy0> &form0,
              const tf::form<Dims, Policy1> &form1) {
  return tf::sqrt(distance2(form0, form1));
}

} // namespace tf
