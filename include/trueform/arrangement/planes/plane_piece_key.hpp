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

#include "../../core/buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::arrangement {

template <typename Index, typename VertexOffsets>
auto plane_piece_key(Index a, Index b, Index n_flat,
                     const VertexOffsets &vertex_offsets)
    -> std::array<Index, 4> {
  const auto endpoint = [&](Index flat) -> std::array<Index, 2> {
    if (flat >= n_flat)
      return {Index(-1), flat - n_flat};
    const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
    return {Index(tag), flat - vertex_offsets[std::size_t(tag)]};
  };
  auto lo = endpoint(a);
  auto hi = endpoint(b);
  if (hi < lo)
    std::swap(lo, hi);
  return {lo[0], lo[1], hi[0], hi[1]};
}

template <typename Index, typename Definition>
auto plane_piece_key(const Definition &definition) -> std::array<Index, 4> {
  return {Index(definition.point_tag_0), definition.point_0,
          Index(definition.point_tag_1), definition.point_1};
}

/// CORE. Restate the rows `[begin, end)` of one plane block in KEY order.
///
/// The block's consumers binary-search it by key — the finalize active branch,
/// the cleanliness oracle — and mint order is not key order once created-first
/// children join, so the block states its keys and the records move by value.
/// The scratch is the caller's, reused across its whole parallel block.
template <typename Index, typename Tier>
auto sort_plane_block_by_key(const Tier &tier, tf::buffer<Index> &rows,
                             std::size_t begin, std::size_t end,
                             tf::buffer<std::array<Index, 5>> &scratch)
    -> void {
  scratch.clear();
  tf::core::reallocate(scratch, end - begin);
  for (auto at = begin; at < end; ++at) {
    const auto key = plane_piece_key<Index>(tier[std::size_t(rows[at])]);
    scratch[at - begin] = {key[0], key[1], key[2], key[3], rows[at]};
  }
  std::sort(scratch.begin(), scratch.end());
  for (auto at = begin; at < end; ++at)
    rows[at] = scratch[at - begin][4];
}

/// The flat piece TICKET a plane block states for one key, `-1` when it
/// states none. A key one block answers twice must answer with one ticket:
/// two would be two identities on one 1-cell.
template <typename Index, typename Tier, typename Block>
auto find_plane_piece_ticket(const Tier &tier, const Block &block,
                             const std::array<Index, 4> &key) -> Index {
  std::size_t lo = 0;
  std::size_t hi = block.size();
  while (lo < hi) {
    const auto mid = lo + (hi - lo) / 2;
    if (plane_piece_key<Index>(tier[std::size_t(block[mid])]) < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == block.size() ||
      plane_piece_key<Index>(tier[std::size_t(block[lo])]) != key)
    return Index(-1);
  const auto ticket = tier.ticket(block[lo]);
  for (auto row = lo + 1; row < block.size() &&
                          plane_piece_key<Index>(tier[std::size_t(
                              block[row])]) == key;
       ++row)
    if (tier.ticket(block[row]) != ticket)
      return Index(-1);
  return ticket;
}

} // namespace tf::arrangement
