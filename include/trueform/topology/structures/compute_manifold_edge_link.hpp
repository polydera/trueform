/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/parallel_apply.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../face_edge_neighbors.hpp"
#include "../face_membership.hpp"
#include "../manifold_edge_peer.hpp"

namespace tf::topology {
template <typename Range0, typename Index, std::size_t N>
auto compute_manifold_edge_link(
    const Range0 &faces, const tf::face_membership<Index> &blink,
    tf::blocked_buffer<manifold_edge_peer<Index>, N> &peer_blocks) {
  auto task_f = [&](Index face_id) {
    auto &&peers = peer_blocks[face_id];
    const auto &face = faces[face_id];
    Index current = N - 1;
    std::array<Index, 2> inner_peers;
    for (Index next = 0; next < Index(N); current = next++) {
      auto it = tf::face_edge_neighbors(blink, faces, face_id,
                                        Index(face[current]), Index(face[next]),
                                        inner_peers.begin(), inner_peers.end());
      switch (it - inner_peers.begin()) {
      case 0:
        peers[current] = {manifold_edge_peer<Index>::boundary};
        break;
      case 1:
        peers[current] = {*it};
        break;
      default:
        peers[current] = {manifold_edge_peer<Index>::non_manifold};
        break;
      }
    }
  };
  tf::parallel_apply(tf::make_sequence_range(faces.size()), task_f);
}
} // namespace tf::topology
