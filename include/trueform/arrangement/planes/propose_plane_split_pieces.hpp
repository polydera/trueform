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

#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_def_respan.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

namespace tf::arrangement {

/// One split parent's whole layout: the current local canonical group a wave
/// cuts and the proposal slots its children own.
///
/// A slot is `first + rank * instances + position` — rank-major in the
/// children's own KEY order, one slot per parent instance, in the parent's
/// span order. That arithmetic is the only correspondence between a parent's
/// rows and its children's, so it is stated once here and read, never
/// restated, by the canonicalize and the block commit.
template <typename Index> struct plane_split_piece_layout {
  Index parent;
  Index first;
  Index pieces;
  Index instances;
};

/// Propose every child instance the wave's splits state — and mint nothing.
///
/// A split is a fact of the canonical GROUP, so it subdivides every instance
/// of it: the parent's span is respanned piece by piece, one child instance
/// per parent instance, every carrier included. A piece states itself through
/// @ref tf::intersect::graph::plane_piece_def, the maker the local
/// arrangement's respan already uses, over the chain the parent's endpoints
/// and its ordered cuts form. A cut is a created identity, so it enters that
/// chain as tag `-1` and its own id.
///
/// A proposal carries `id == -1`: identity is the canonicalize's to assign,
/// after equal keys have met. Everything else the record states is final —
/// the provenance verbatim off the parent instance, the endpoints and flags
/// off the piece maker.
///
/// `carrier_planes` is the parents' carriers, the closure retirement depends
/// on: every one of them already holds a local block, or the call refuses.
///
/// False rejects the call and publishes nothing: the local tier's interior
/// was proven by the producers that appended it, so this states only what the
/// emission itself indexes.
template <typename Index, typename Int, typename PlaneOfFace>
auto propose_plane_split_pieces(
    const tf::buffer<Index> &split_parents,
    const tf::buffer<Index> &split_offsets, const tf::buffer<Index> &split_data,
    const PlaneOfFace &plane_of_face,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket,
    tf::buffer<plane_split_piece_layout<Index>> &layout,
    tf::buffer<tf::intersect::graph::plane_edge_def<Index>> &proposals,
    tf::buffer<Index> &carrier_planes) -> bool {
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  const auto n_parents = split_parents.size();
  const auto n_canon = local_tables.n_canon();
  const auto n_blocks = local_tables.edges().size();
  if (split_offsets.size() == 0) {
    if (n_parents != 0 || split_data.size() != 0)
      return false;
  } else if (split_offsets.size() != n_parents + 1 ||
             split_offsets[0] != Index(0) ||
             split_offsets[n_parents] != Index(split_data.size()))
    return false;
  if (!local_tables.well_formed())
    return false;
  const auto &def_offsets = local_tables.def_offsets();
  if (tf::parallel_contains(
          tf::make_sequence_range(n_parents),
          [&](std::size_t at) {
            const auto parent = split_parents[at];
            return parent < Index(0) || parent >= n_canon ||
                   (at != 0 && split_parents[at - 1] >= parent) ||
                   split_offsets[at + 1] <= split_offsets[at] ||
                   def_offsets[std::size_t(parent) + 1] <=
                       def_offsets[std::size_t(parent)];
          },
          tf::checked))
    return false;
  // a cut is a created identity, so the probe it answers is an id
  if (tf::parallel_contains(
          tf::make_range(split_data), [](Index cut) { return cut < Index(0); },
          tf::checked))
    return false;

  layout.clear();
  proposals.clear();
  carrier_planes.clear();
  if (n_parents == 0)
    return true;

  if (tf::parallel_contains(
          tf::make_range(split_parents),
          [&](Index parent) {
            for (const auto &def : local_tables.canon_group(parent)) {
              const auto plane = plane_of_face(def.face);
              if (plane < Index(0) ||
                  std::size_t(plane) >= plane_ticket.size())
                return true;
              const auto block = plane_ticket[std::size_t(plane)];
              if (block < Index(0) || std::size_t(block) >= n_blocks)
                return true;
            }
            return false;
          },
          tf::checked))
    return false;
  // the carriers are a SET over the dense plane space: the mask states
  // membership and the compaction states it ascending
  tf::buffer<char> carries;
  carries.allocate_and_initialize(plane_ticket.size(), char(0));
  tf::parallel_for_each(
      tf::make_range(split_parents),
      [&](Index parent) {
        for (const auto &def : local_tables.canon_group(parent))
          carries[std::size_t(plane_of_face(def.face))] = 1;
      },
      tf::checked);
  tf::sequenced_generate(
      tf::make_sequence_range(plane_ticket.size()), carrier_planes,
      [&carries](std::size_t plane, tf::buffer<Index> &out) {
        if (carries[plane] != 0)
          out.push_back(Index(plane));
      },
      tf::checked);

  const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
  layout.allocate(n_parents);
  std::size_t emitted = 0;
  for (std::size_t at = 0; at < n_parents; ++at) {
    const auto parent = std::size_t(split_parents[at]);
    const auto pieces =
        std::size_t(split_offsets[at + 1] - split_offsets[at]) + 1;
    const auto instances =
        std::size_t(def_offsets[parent + 1] - def_offsets[parent]);
    if (pieces > index_extent / instances ||
        pieces * instances > index_extent - emitted) {
      layout.clear();
      carrier_planes.clear();
      return false;
    }
    layout[at] = {split_parents[at], Index(emitted), Index(pieces),
                  Index(instances)};
    emitted += pieces * instances;
  }
  proposals.allocate(emitted);

  struct local_t {
    tf::buffer<std::array<Index, 2>> chain;
    tf::buffer<def_t> probe;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(n_parents),
      [&](std::size_t at, local_t &local) {
        const auto &split = layout[at];
        const auto span = local_tables.canon_group(split.parent);
        const auto cuts = std::size_t(split.pieces) - 1;
        const auto splits = std::size_t(split_offsets[at]);
        const auto &first = span[0];
        local.chain.allocate(cuts + 2);
        local.chain[0] = {Index(first.point_tag_0), first.point_0};
        for (std::size_t k = 0; k < cuts; ++k)
          local.chain[k + 1] = {Index(-1), split_data[splits + k]};
        local.chain[cuts + 1] = {Index(first.point_tag_1), first.point_1};
        // the piece maker states a piece's key, so the children's own order
        // is read off it and never restated
        local.probe.allocate(std::size_t(split.pieces));
        for (Index k = 0; k < split.pieces; ++k) {
          auto piece = first;
          tf::intersect::graph::plane_piece_def(
              piece, local.chain[std::size_t(k)],
              local.chain[std::size_t(k) + 1], first, split.pieces);
          piece.id = k;
          local.probe[std::size_t(k)] = piece;
        }
        std::sort(local.probe.begin(), local.probe.end(),
                  [](const def_t &x, const def_t &y) {
                    return tf::intersect::graph::plane_def_key_less(x, y) ||
                           (!tf::intersect::graph::plane_def_key_less(y, x) &&
                            x.id < y.id);
                  });
        auto write = std::size_t(split.first);
        for (Index rank = 0; rank < split.pieces; ++rank) {
          const auto k = std::size_t(local.probe[std::size_t(rank)].id);
          for (const auto &instance : span) {
            auto piece = instance;
            tf::intersect::graph::plane_piece_def(
                piece, local.chain[k], local.chain[k + 1], instance,
                split.pieces);
            piece.id = Index(-1);
            proposals[write++] = piece;
          }
        }
      },
      local_t{});

  return true;
}

} // namespace tf::arrangement
