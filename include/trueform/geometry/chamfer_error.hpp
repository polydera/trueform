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

#include "../core/algorithm/reduce.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/frame_of.hpp"
#include "../core/sqrt.hpp"
#include "../core/transformed.hpp"
#include "../spatial/neighbor_search.hpp"
#include "../spatial/policy/tree.hpp"

#include <type_traits>

namespace tf {

/// @ingroup geometry_registration
/// @brief Compute one-way Chamfer error from A to B (mean nearest-neighbor distance).
///
/// For each point in A, finds the nearest point in B and accumulates the distance.
/// Returns the mean distance. This is an asymmetric measure; for symmetric Chamfer
/// distance, compute both directions and average.
///
/// If point sets have frames attached, the computation is performed in world space.
///
/// @param A Source point set.
/// @param B Target point set (must have tree policy for efficient search).
/// @return Mean nearest-neighbor distance from A to B.
template <typename Policy0, typename Policy1>
auto chamfer_error(const tf::points<Policy0> &A, const tf::points<Policy1> &B) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Target point set B must have a tree policy attached");

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
}

} // namespace tf
