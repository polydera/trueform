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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/checked.hpp"
#include "../core/constants.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/frame_of.hpp"
#include "../core/point.hpp"
#include "../core/policy/frame.hpp"
#include "../core/polygons.hpp"
#include "../core/small_vector.hpp"
#include "../core/transformed.hpp"
#include "../core/views/zip.hpp"
#include "./mod_tree_like.hpp"
#include "./policy/tree.hpp"
#include "./policy/winding.hpp"
#include "./winding/traverse.hpp"
#include "./winding/winding_tree.hpp"
#include <cstddef>
#include <type_traits>

namespace tf {

namespace spatial {
template <typename Policy>
auto winding_is_mod_tree(const tf::mod_tree_like<Policy> *) -> std::true_type;
auto winding_is_mod_tree(const void *) -> std::false_type;
} // namespace spatial

/// @ingroup spatial_queries
/// @brief Accuracy knob of the winding queries: a node is evaluated by
/// its stored far-field expansion when the query is farther than `beta`
/// times the node's far-field radius. Larger is more accurate and descends
/// deeper; the exact sum is the limit.
struct winding_config {
  double beta = 2.0;
};

/// @ingroup spatial_queries
/// @brief The generalized winding number of a polygon form at a point.
///
/// Approximately 1 inside a closed surface, 0 outside, and a graceful
/// fractional value for open sheets and soups. Near the surface the
/// value crosses 0.5 continuously, so any threshold on it is a float
/// test — closed-input signing belongs to the exact predicates, this
/// query to the inputs they refuse.
///
/// Requires `polygons | tf::tag(tree) | tf::tag(winding_moments)`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto winding_number(const tf::polygons<Policy0> &polygons,
                    const tf::point_like<Dims, Policy1> &point,
                    winding_config config = {}) -> double {
  static_assert(Dims == 3, "winding_number requires 3D");
  static_assert(tf::coordinate_dims_v<Policy0> == Dims,
                "Polygon dimensions must match point dimensions");
  static_assert(tf::has_tree_policy<Policy0>, "Use polygons | tf::tag(tree)");
  static_assert(tf::has_winding_policy<Policy0>,
                "Use polygons | tf::tag(winding_moments)");
  static_assert(
      !decltype(spatial::winding_is_mod_tree(
          static_cast<const std::decay_t<decltype(polygons.tree())> *>(
              nullptr)))::value,
      "winding_number requires a static tree: build the moments over "
      "tf::tree");
  const auto &tree = polygons.tree();
  const auto winding =
      spatial::make_winding_tree(tree, polygons.winding_moments());
  const auto q =
      tf::transformed(point.template as<double>(),
                      tf::frame_of(polygons).inverse_transformation());
  tf::small_vector<typename std::decay_t<decltype(tree)>::index_type, 512>
      stack;
  const auto inv_4pi = 0.25 / tf::pi<double>;
  return inv_4pi * spatial::winding_traverse(winding, polygons, q,
                                             config.beta * config.beta, stack);
}

/// @ingroup spatial_queries
/// @brief The generalized winding numbers of a polygon form at every
/// point of a range, written to `out` in the same order.
template <typename Policy0, typename Points, typename Out>
auto winding_number(const tf::polygons<Policy0> &polygons, const Points &points,
                    Out &&out, winding_config config = {}) -> void {
  static_assert(tf::coordinate_dims_v<Policy0> == 3,
                "winding_number requires 3D");
  static_assert(tf::has_tree_policy<Policy0>, "Use polygons | tf::tag(tree)");
  static_assert(tf::has_winding_policy<Policy0>,
                "Use polygons | tf::tag(winding_moments)");
  static_assert(
      !decltype(spatial::winding_is_mod_tree(
          static_cast<const std::decay_t<decltype(polygons.tree())> *>(
              nullptr)))::value,
      "winding_number requires a static tree: build the moments over "
      "tf::tree");
  const auto &tree = polygons.tree();
  using index_t = typename std::decay_t<decltype(tree)>::index_type;
  const auto winding =
      spatial::make_winding_tree(tree, polygons.winding_moments());
  const auto pose = tf::frame_of(polygons);
  const auto beta2 = config.beta * config.beta;
  struct local_t {
    tf::small_vector<index_t, 512> stack;
  };
  tf::parallel_for_each(
      tf::zip(points, out),
      [&winding, &polygons, &pose, beta2](auto pair, local_t &local) {
        auto &&[query, value] = pair;
        const auto inv_4pi = 0.25 / tf::pi<double>;
        const auto q = tf::transformed(query.template as<double>(),
                                       pose.inverse_transformation());
        value = std::decay_t<decltype(value)>(
            inv_4pi * spatial::winding_traverse(winding, polygons, q, beta2,
                                                local.stack));
      },
      local_t{}, tf::checked(8));
}

} // namespace tf
