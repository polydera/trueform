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
#include "../core/polygons.hpp"
#include "../core/static_size.hpp"
#include "../core/views/mapped_range.hpp"
#include "./edge_representation.hpp"
#include "./face_membership.hpp"
#include "./manifold_edge_link.hpp"
#include "./policy/manifold_edge_link.hpp"
#include <functional>
#include <type_traits>

namespace tf {

/// @ingroup topology_analysis
/// @brief Computes the Euler characteristic of a polygon mesh.
///
/// The Euler characteristic is defined as V - E + F, where V is the number
/// of vertices, E is the number of unique edges, and F is the number of
/// faces. Each undirected edge is counted once, by the face the manifold
/// edge link makes its representative, so boundary and non-manifold edges
/// count exactly like interior ones.
///
/// Builds manifold edge link internally if not provided via policy.
///
/// @tparam Policy The polygons policy.
/// @param polygons The polygon collection.
/// @return The Euler characteristic (V - E + F).
template <typename Policy>
auto euler_characteristic(const tf::polygons<Policy> &polygons) -> int {
  int V = polygons.points().size();
  int F = polygons.faces().size();

  auto count_edges = [](const auto &link) {
    return tf::reduce(
        tf::make_mapped_range(tf::make_edge_representation(link),
                              [](const auto &face_edges) {
                                int count = 0;
                                for (bool represents : face_edges)
                                  count += represents;
                                return count;
                              }),
        std::plus<>{}, int{0}, tf::checked);
  };

  if constexpr (tf::has_manifold_edge_link_policy<Policy>)
    return V - count_edges(polygons.manifold_edge_link()) + F;
  else {
    using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
    tf::face_membership<Index> fm;
    fm.build(polygons);
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(polygons.faces()[0])>>
        mel;
    mel.build(polygons.faces(), fm);
    return V - count_edges(mel) + F;
  }
}

} // namespace tf
