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
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../manifold_facet_peer.hpp"
#include "../tet_facet_neighbors.hpp"
#include "../tet_membership_like.hpp"
#include <algorithm>
#include <iterator>
#include <type_traits>

namespace tf::topology {
template <typename Index, typename Range0, typename Policy, typename Range1>
auto compute_manifold_facet_link(const Range0 &tetras,
                                 const tf::tet_membership_like<Policy> &blink,
                                 Range1 &peer_blocks) {
  auto task_f = [&](auto begin, auto end) {
    tf::small_vector<Index, 4> inner_peers;
    while (begin != end) {
      Index tet_id = *begin++;
      auto &&peers = peer_blocks[tet_id];
      const auto &tet = tetras[tet_id];
      for (Index slot = 0; slot < 4; ++slot) {
        inner_peers.clear();
        tf::tet_facet_neighbors(blink, tet_id, Index(tet[(slot + 1) & 3]),
                                Index(tet[(slot + 2) & 3]),
                                Index(tet[(slot + 3) & 3]),
                                std::back_inserter(inner_peers));
        switch (inner_peers.size()) {
        case 0:
          peers[slot] = {manifold_facet_peer<Index>::boundary};
          break;
        case 1:
          peers[slot] = {inner_peers[0]};
          break;
        default:
          if (std::all_of(inner_peers.begin(), inner_peers.end(),
                          [&](const auto &x) { return x > tet_id; }))
            peers[slot] = {
                manifold_facet_peer<Index>::non_manifold_representative};
          else
            peers[slot] = {manifold_facet_peer<Index>::non_manifold};
          break;
        }
      }
    }
  };
  tf::parallel_for(tf::make_sequence_range(tetras.size()), task_f);
}
template <typename Range0, typename Policy, typename Range1>
auto compute_manifold_facet_link(const Range0 &tetras,
                                 const tf::tet_membership_like<Policy> &blink,
                                 Range1 &peer_blocks) {
  using Index = std::decay_t<decltype(peer_blocks[0][0].tet_peer)>;
  return compute_manifold_facet_link<Index>(tetras, blink, peer_blocks);
}
} // namespace tf::topology
