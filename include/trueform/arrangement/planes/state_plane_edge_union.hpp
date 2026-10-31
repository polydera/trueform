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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/plane_identity_names.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_flat_weld.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_recovery_statement.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::arrangement {

/// One canonical edge holds one split set on every carrier. In: the keys of
/// the `dirty_planes`' rows the merges moved; out: the landings as
/// `statements` and `topology`, the identities one parameter places as `welds`.
template <typename Index, typename Param, typename World, typename Int,
          typename VertexOffsets>
auto state_plane_edge_union(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket,
    const tf::buffer<Index> &dirty_planes,
    const tf::buffer<Index> &created_class, const tf::buffer<Param> &class_t,
    const tf::buffer<plane_recovery_name<Index, Param>> &class_name,
    const VertexOffsets &vertex_offsets, Index n_flat, Index base_created,
    Index crossing_name_kind, Index refinement_name_kind, Param whole,
    tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    tf::buffer<plane_recovery_proposal<Index, Param>> &topology,
    tf::buffer<plane_flat_weld<Index>> &welds) -> void {
  using key_t = std::array<Index, 2>;
  struct piece_t {
    Index key;
    Index group;
    Param s0;
    Param s1;
    Index plane;
    std::uint8_t inverted;
  };
  struct identity_t {
    Index key;
    Param s;
    Index flat;
  };
  if (plane_ticket.size() == 0)
    return;
  const auto merges = world.merges();
  const auto created_base = n_flat + base_created;
  const auto merged = [&](Index flat) {
    if (flat >= created_base)
      return flat;
    const bool created = flat >= n_flat;
    const auto target = tf::intersect::graph::plane_merge_target(
        merges, Index(created), created ? flat - n_flat : flat);
    return target == Index(-1) ? flat : n_flat + target;
  };
  // the home edge a created identity's name places it on, as the merged key
  // and the parameter along it; a key of -1 when it names none
  const auto home = [&](Index flat, Param &s, bool &moved) -> key_t {
    const auto cls = created_class[std::size_t(flat - created_base)];
    if (cls < Index(0))
      return {Index(-1), Index(-1)};
    const auto &name = class_name[std::size_t(cls)];
    const auto kind = name.feature[0];
    if ((kind != crossing_name_kind && kind != refinement_name_kind) ||
        name.feature[1] >= created_base || name.feature[2] >= created_base)
      return {Index(-1), Index(-1)};
    const auto a = merged(name.feature[1]);
    const auto b = merged(name.feature[2]);
    moved = moved || a != name.feature[1] || b != name.feature[2];
    const auto t = class_t[std::size_t(cls)];
    s = a < b ? t : Param(whole - t);
    return {std::min(a, b), std::max(a, b)};
  };
  // the edge a row lies on: the home its created ends share, or its own
  // merged ends; `moved` says the merges moved an end or a home
  const auto edge_of = [&](const std::array<Index, 2> &ends,
                           bool &moved) -> key_t {
    key_t key{Index(-1), Index(-1)};
    for (const auto end : ends) {
      if (end < created_base) {
        moved = moved || merged(end) != end;
        continue;
      }
      Param s;
      const auto placed = home(end, s, moved);
      if (placed[0] == Index(-1) || (key[0] != Index(-1) && key != placed))
        return {Index(-1), Index(-1)};
      key = placed;
    }
    if (key[0] != Index(-1))
      return key;
    const auto a = merged(ends[0]);
    const auto b = merged(ends[1]);
    return {std::min(a, b), std::max(a, b)};
  };
  const auto blocks = tf::make_range(local_tables.edges());
  const auto defs = local_tables.edge_defs();
  // the rows of the planes' blocks laid end to end: one carrier can hold
  // nearly every row a sweep reads, so a row is the parallel grain
  const auto row_offsets_of = [&](const tf::buffer<Index> &planes) {
    tf::buffer<Index> offsets;
    offsets.allocate(planes.size() + 1);
    offsets[0] = Index(0);
    for (std::size_t at = 0; at < planes.size(); ++at) {
      const auto block = plane_ticket[std::size_t(planes[at])];
      offsets[at + 1] =
          offsets[at] + (block == Index(-1)
                             ? Index(0)
                             : Index(blocks[std::size_t(block)].size()));
    }
    return offsets;
  };
  const auto carrier_of_row = [](const tf::buffer<Index> &offsets, Index at) {
    return std::size_t(
        std::upper_bound(offsets.begin() + 1, offsets.end(), at) -
        (offsets.begin() + 1));
  };
  const auto row_at = [&](const tf::buffer<Index> &planes,
                          const tf::buffer<Index> &offsets, std::size_t carrier,
                          Index at) {
    const auto block = plane_ticket[std::size_t(planes[carrier])];
    return blocks[std::size_t(block)][std::size_t(at - offsets[carrier])];
  };

  const auto dirty_rows = row_offsets_of(dirty_planes);
  tf::buffer<key_t> keys;
  tf::generic_generate(
      tf::make_sequence_range(dirty_rows[dirty_planes.size()]), keys,
      [&](Index at, tf::buffer<key_t> &out) {
        const auto row = row_at(dirty_planes, dirty_rows,
                                carrier_of_row(dirty_rows, at), at);
        if (row < Index(0))
          return;
        bool moved = false;
        const auto key =
            edge_of(plane_recovery_flat_edge_ends(defs[std::size_t(row)],
                                                  n_flat, vertex_offsets),
                    moved);
        if (moved && key[0] != Index(-1))
          out.push_back(key);
      },
      tf::checked);
  if (keys.size() == 0)
    return;
  tbb::parallel_sort(keys.begin(), keys.end());
  keys.erase_till_end(std::unique(keys.begin(), keys.end()));

  // an edge's carriers are the dirty planes and the instances of the one
  // world group its key names
  const auto pair_of_flat = [&](Index flat) -> std::array<Index, 2> {
    if (flat >= n_flat)
      return {Index(-1), flat - n_flat};
    const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
    return {Index(tag), flat - Index(vertex_offsets[std::size_t(tag)])};
  };
  tf::buffer<Index> carriers;
  tf::core::append(dirty_planes, carriers);
  for (const auto &key : keys) {
    auto lo = pair_of_flat(key[0]);
    auto hi = pair_of_flat(key[1]);
    if (hi < lo)
      std::swap(lo, hi);
    tf::intersect::graph::plane_edge_def<Index> probe{};
    probe.point_tag_0 = std::int16_t(lo[0]);
    probe.point_0 = lo[1];
    probe.point_tag_1 = std::int16_t(hi[0]);
    probe.point_1 = hi[1];
    const auto group =
        tf::intersect::graph::find_plane_canon_group(world.tables(), probe);
    if (group != Index(-1))
      for (const auto &instance : world.canon_group(group))
        carriers.push_back(world.plane_of_face(instance.face));
  }
  std::sort(carriers.begin(), carriers.end());
  carriers.erase_till_end(std::unique(carriers.begin(), carriers.end()));

  const auto carrier_rows = row_offsets_of(carriers);
  tf::buffer<piece_t> pieces;
  tf::buffer<identity_t> identities;
  tf::generic_generate(
      tf::make_sequence_range(carrier_rows[carriers.size()]),
      std::tie(pieces, identities),
      [&](Index at, auto &out) {
        auto &piece_out = std::get<0>(out);
        auto &identity_out = std::get<1>(out);
        const auto carrier = carrier_of_row(carrier_rows, at);
        const auto row = row_at(carriers, carrier_rows, carrier, at);
        if (row < Index(0))
          return;
        const auto plane = carriers[carrier];
        const auto &def = defs[std::size_t(row)];
        const auto ends =
            plane_recovery_flat_edge_ends(def, n_flat, vertex_offsets);
        bool moved = false;
        const auto edge = edge_of(ends, moved);
        const auto found = std::lower_bound(keys.begin(), keys.end(), edge);
        if (found == keys.end() || *found != edge)
          return;
        const auto k = Index(found - keys.begin());
        const auto on_edge = [&](Index flat, Param &s) {
          if (flat == edge[0])
            s = Param(0);
          else if (flat == edge[1])
            s = whole;
          else if (flat >= created_base) {
            bool unread = false;
            home(flat, s, unread);
            identity_out.push_back({k, s, flat});
          } else
            return false;
          return true;
        };
        Param s0, s1;
        if (!on_edge(merged(ends[0]), s0) || !on_edge(merged(ends[1]), s1) ||
            s0 == s1)
          return;
        piece_out.push_back(
            {k, def.id, s0, s1, plane,
             std::uint8_t(plane_recovery_flat_edge_inverted(ends))});
      },
      tf::checked);
  tbb::parallel_sort(
      pieces.begin(), pieces.end(), [](const piece_t &x, const piece_t &y) {
        return std::tie(x.key, x.group, x.plane) <
               std::tie(y.key, y.group, y.plane);
      });
  pieces.erase_till_end(std::unique(
      pieces.begin(), pieces.end(), [](const piece_t &x, const piece_t &y) {
        return x.key == y.key && x.group == y.group;
      }));
  tbb::parallel_sort(identities.begin(), identities.end(),
                     [](const identity_t &x, const identity_t &y) {
                       return std::tie(x.key, x.s, x.flat) <
                              std::tie(y.key, y.s, y.flat);
                     });
  identities.erase_till_end(
      std::unique(identities.begin(), identities.end(),
                  [](const identity_t &x, const identity_t &y) {
                    return x.key == y.key && x.flat == y.flat;
                  }));

  // one parameter is one identity: the first of a run speaks for the rest
  tf::buffer<identity_t> placed;
  for (std::size_t at = 0; at < identities.size(); ++at) {
    const auto &identity = identities[at];
    if (placed.size() != 0 && placed.back().key == identity.key &&
        placed.back().s == identity.s) {
      welds.push_back({placed.back().flat, identity.flat, Index(-1)});
      continue;
    }
    placed.push_back(identity);
  }
  // the pieces state the landings in their own definition order, each
  // parameter in the name frame the wave mirrors back
  for (const auto &piece : pieces) {
    const auto lo = std::min(piece.s0, piece.s1);
    const auto hi = std::max(piece.s0, piece.s1);
    auto at = std::upper_bound(
        placed.begin(), placed.end(), std::make_pair(piece.key, lo),
        [](const std::pair<Index, Param> &value, const identity_t &identity) {
          return std::tie(value.first, value.second) <
                 std::tie(identity.key, identity.s);
        });
    for (; at != placed.end() && at->key == piece.key && at->s < hi; ++at) {
      const auto t = piece.s0 < piece.s1 ? at->s : Param(whole - at->s);
      const auto parameter = piece.inverted ? Param(whole - t) : t;
      const auto name = make_plane_recovery_name<Index, Param>(
          tf::intersect::graph::plane_point_name<Index>(at->flat - n_flat));
      statements.push_back({name, piece.group, piece.plane, Index(1)});
      topology.push_back({name, piece.group, piece.plane, Index(1), parameter,
                          std::uint8_t(1)});
    }
  }
}

} // namespace tf::arrangement
