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

#include "../core/algorithm/parallel_transform.hpp"
#include "../core/algorithm/reduce.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/frame_of.hpp"
#include "../core/sqrt.hpp"
#include "../core/transformed.hpp"
#include "../core/views/take.hpp"
#include "../spatial/neighbor_search.hpp"
#include "../spatial/policy/tree.hpp"
#include "tbb/parallel_sort.h"

#include <type_traits>

namespace tf {

namespace geometry {
template <typename Policy0, typename Policy1>
auto chamfer_error(const tf::points<Policy0> &A, const tf::points<Policy1> &B,
                   float outlier_proportion,
                   tf::buffer<tf::coordinate_type<Policy0, Policy1>> &buffer) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Target point set B must have a tree policy attached");

  using T = tf::coordinate_type<Policy0, Policy1>;
  buffer.allocate(A.size());
  tf::parallel_transform(
      A, buffer,
      [&](const auto &arg) {
        auto query = tf::transformed(arg, tf::frame_of(A));
        auto [id, cpt] = tf::neighbor_search(B, query);
        return tf::sqrt(cpt.metric);
      },
      tf::checked);
  tbb::parallel_sort(buffer);
  auto size = buffer.size() - std::size_t(buffer.size() * outlier_proportion);
  return tf::reduce(tf::take(buffer, size), std::plus<>{}, T(0), tf::checked) /
         size;
}
} // namespace geometry

/// @ingroup geometry_registration
/// @brief Compute one-way Chamfer error from A to B (mean nearest-neighbor
/// distance).
///
/// For each point in A, finds the nearest point in B and accumulates the
/// distance. Returns the mean distance. This is an asymmetric measure; for
/// symmetric Chamfer distance, compute both directions and average.
///
/// If point sets have frames attached, the computation is performed in world
/// space.
///
/// @param A Source point set.
/// @param B Target point set (must have tree policy for efficient search).
/// @return Mean nearest-neighbor distance from A to B.
template <typename Policy0, typename Policy1>
auto chamfer_error(const tf::points<Policy0> &A, const tf::points<Policy1> &B,
                   float outlier_proportion = 0) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Target point set B must have a tree policy attached");

  if (outlier_proportion <= 0 || outlier_proportion >= 1) {
    using T = tf::coordinate_type<Policy0, Policy1>;
    auto sum = tf::reduce(
        A,
        [&](T acc, const auto &arg) {
          if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, T>) {
            return acc + arg;
          } else {
            auto query = tf::transformed(arg, tf::frame_of(A));
            auto [id, cpt] = tf::neighbor_search(B, query);
            return acc + tf::sqrt(cpt.metric);
          }
        },
        T(0), tf::checked);

    return sum / T(A.size());
  } else {
    tf::buffer<tf::coordinate_type<Policy0, Policy1>> buffer;
    return geometry::chamfer_error(A, B, outlier_proportion, buffer);
  }
}

} // namespace tf
