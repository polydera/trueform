/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/algorithm/reduce.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/face_edge_neighbors.hpp"
#include "../../topology/manifold_edge_peer.hpp"
#include "./count_tet_open_edges.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf::tetmesh {

/// The edges the triangles `kept` a facet merge (@ref merge_tet_facets) left
/// of `triangles` leave open (@ref tet_open_edges): read off the peers
/// `link(t)` names when the merge threw nothing out, counted on `steps`
/// otherwise.
template <typename Triangles, typename Membership, typename Link,
          typename Index>
auto count_tet_merged_open_edges(const Triangles &triangles,
                                 const Membership &membership, const Link &link,
                                 const tf::buffer<std::array<Index, 3>> &kept,
                                 const std::array<std::size_t, 3> &thrown,
                                 tf::buffer<tet_edge_step<Index>> &steps)
    -> tet_open_edges {
  using peer_t = tf::manifold_edge_peer<Index>;
  if (thrown[0] || thrown[1])
    return count_tet_open_edges(kept, steps);
  const auto corners = [&triangles](std::size_t t) {
    const auto &triangle = triangles[t];
    return std::array<Index, 3>{Index(triangle[0]), Index(triangle[1]),
                                Index(triangle[2])};
  };
  const auto runs = [&corners](std::size_t t, Index u, Index v) {
    const auto c = corners(t);
    for (std::size_t k = 0; k < 3; ++k)
      if (c[k] == u && c[k == 2 ? 0 : k + 1] == v)
        return true;
    return false;
  };
  return tf::reduce(
      tf::make_sequence_range(triangles.size()),
      [&link, &membership, &triangles, &corners, &runs](tet_open_edges open,
                                                        const auto &at) {
        if constexpr (std::is_same_v<std::decay_t<decltype(at)>,
                                     tet_open_edges>) {
          open.unbalanced += at.unbalanced;
          open.boundary += at.boundary;
          return open;
        } else {
          const std::size_t t = std::size_t(at);
          const auto c = corners(t);
          const auto peers = link(t);
          for (std::size_t k = 0; k < 3; ++k) {
            const Index p = peers[k].face_peer;
            const Index u = c[k], v = c[k == 2 ? 0 : k + 1];
            if (p == peer_t::boundary) {
              ++open.unbalanced;
              ++open.boundary;
              continue;
            }
            if (p >= 0) {
              open.unbalanced +=
                  std::size_t(p) > t && runs(std::size_t(p), u, v);
              continue;
            }
            if (p != peer_t::non_manifold_representative)
              continue;
            int balance = 1;
            tf::face_edge_neighbors_apply(
                membership, triangles, Index(t), u, v, [&](Index r) {
                  balance += runs(std::size_t(r), u, v) ? 1 : -1;
                  return false;
                });
            open.unbalanced += balance != 0;
          }
          return open;
        }
      },
      tet_open_edges{});
}

} // namespace tf::tetmesh
