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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/plane_identity_names.hpp"
#include "./plane_recovery_name.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::arrangement {

template <typename Index> using plane_birth_edge_record = std::array<Index, 5>;

/// `tier` says which table indexes `root` — `0` original, `1` local — the
/// same pair-name every statement speaks. A producer marks its local birth
/// candidate; the class barrier elects one after equal names meet.
template <typename Index, typename Param> struct plane_recovery_proposal {
  plane_recovery_name<Index, Param> name;
  Index root, plane, tier;
  Param parameter;
  std::uint8_t birth;
};

template <typename Index, typename Param>
auto close_plane_recovery_proposals(
    tf::buffer<plane_recovery_proposal<Index, Param>> &proposals,
    tf::buffer<Index> &class_offsets) -> Index {
  using proposal_t = plane_recovery_proposal<Index, Param>;
  class_offsets.clear();
  if (proposals.size() == 0)
    return Index(0);
  tbb::parallel_sort(
      proposals.begin(), proposals.end(),
      [](const proposal_t &x, const proposal_t &y) {
        return std::tie(x.name, x.tier, x.root, x.plane, x.parameter,
                        x.birth) <
               std::tie(y.name, y.tier, y.root, y.plane, y.parameter,
                        y.birth);
      });
  class_offsets.reserve(proposals.size() + 1);
  tf::sequenced_generate(
      tf::make_sequence_range(proposals.size()), class_offsets,
      [&proposals](std::size_t at, tf::buffer<Index> &out) {
        if (at == 0 || !(proposals[at - 1].name == proposals[at].name))
          out.push_back(Index(at));
      },
      tf::checked);
  class_offsets.push_back(Index(proposals.size()));
  return Index(class_offsets.size()) - Index(1);
}

/// A definition's endpoint flats in its OWN key order.
template <typename Index, typename VertexOffsets>
auto plane_recovery_flat_edge_ends(
    const tf::intersect::graph::plane_edge_def<Index> &def, Index n_flat,
    const VertexOffsets &vertex_offsets) -> std::array<Index, 2> {
  const auto flat = [&](std::int16_t tag, Index id) {
    return tag < 0 ? n_flat + id : vertex_offsets[std::size_t(tag)] + id;
  };
  return {flat(def.point_tag_0, def.point_0),
          flat(def.point_tag_1, def.point_1)};
}

/// THE FRAME LAW: a parameter a NAME carries runs `feature[0] -> feature[1]`,
/// the definition's endpoint flats ascending. True when the definition's own
/// order runs against that — a created endpoint keys FIRST, as tag `-1`, and
/// flattens LAST, past every original vertex — so a definition-frame parameter
/// on that edge is the mirror of the one its name states.
template <typename Index>
auto plane_recovery_flat_edge_inverted(const std::array<Index, 2> &ends)
    -> bool {
  return ends[1] < ends[0];
}

template <typename Index, typename VertexOffsets>
auto plane_recovery_flat_edge_key(
    const tf::intersect::graph::plane_edge_def<Index> &def, Index n_flat,
    const VertexOffsets &vertex_offsets) -> std::array<Index, 2> {
  auto key = plane_recovery_flat_edge_ends(def, n_flat, vertex_offsets);
  if (plane_recovery_flat_edge_inverted(key))
    std::swap(key[0], key[1]);
  return key;
}

/// Order the identities a recovery wave places on each exact current piece.
///
/// The CDT parameter is the transportable split fact: after a shared identity
/// is born on one piece, its lattice position may bend the other piece, so
/// projecting that elected position back onto the old segment would discard
/// the split that creates the bend. Repeated observations of one
/// `{piece, identity}` retain the lowest parameter, then the compact records
/// are ordered once in piece/parameter order for respan.
///
/// `parameters_in_name_frame` says the proposals speak THE FRAME LAW — the
/// name's frame — so an inverted definition's cuts order by the mirrored
/// parameter and the chain the respan builds, `[point_0, cuts..., point_1]`,
/// stays the definition's own. The identities are untouched; only their order
/// on one edge is. A producer that binds its parameter to the edge instead of
/// the name passes false.
template <typename Index, typename Param, typename Graph, typename Tables,
          typename VertexOffsets>
auto order_plane_recovery_splits(
    const Graph &graph, const Tables &local_tables,
    const tf::buffer<plane_recovery_proposal<Index, Param>> &proposals,
    const tf::buffer<Index> &class_offsets, const tf::buffer<Index> &class_id,
    const tf::buffer<std::array<Index, 3>> &merges,
    const VertexOffsets &vertex_offsets, Index n_classes, Param whole,
    bool parameters_in_name_frame, tf::buffer<Index> &split_edge,
    tf::buffer<Index> &split_tier, tf::buffer<Index> &split_offsets,
    tf::buffer<Index> &split_data, std::size_t &out_of_span,
    std::size_t &on_endpoint) -> void {
  struct cut_t {
    Index tier;
    Index edge;
    Param parameter;
    Index id;
  };
  const auto n_flat = Index(vertex_offsets[vertex_offsets.size() - 1]);
  const auto n_proposed =
      n_classes == Index(0)
          ? std::size_t(0)
          : std::size_t(class_offsets[std::size_t(n_classes)]);
  // Every proposal states at most one cut, so each is answered in its own slot
  // and its verdict rides beside it; the census reasons and the kept extent
  // then come out of one adjacent sweep that also compacts.
  const char verdict_kept = 0;
  const char verdict_on_endpoint = 1;
  const char verdict_out_of_span = 2;
  const char verdict_unnamed = 3;
  tf::buffer<cut_t> cuts;
  tf::buffer<char> verdict;
  cuts.allocate(n_proposed);
  verdict.allocate(n_proposed);
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t(n_classes)),
      [&](std::size_t cls) {
        const auto id = class_id[cls];
        const auto begin = std::size_t(class_offsets[cls]);
        const auto end = std::size_t(class_offsets[cls + 1]);
        for (auto i = begin; i < end; ++i) {
          if (id == Index(-1)) {
            verdict[i] = verdict_unnamed;
            continue;
          }
          const auto &proposal = proposals[i];
          const auto &def = proposal.tier == Index(0)
                                ? graph.canon_group(proposal.root)[0]
                                : local_tables.canon_group(proposal.root)[0];
          const auto ends = tf::intersect::graph::plane_merged_endpoints(
              merges, vertex_offsets, def);
          if ((ends.lo_tag < 0 && ends.lo_id == id) ||
              (ends.hi_tag < 0 && ends.hi_id == id)) {
            verdict[i] = verdict_on_endpoint;
            continue;
          }
          // THE ABSOLUTE: A SPLIT THIS TIER DEMANDED IS NEVER DROPPED IN
          // SILENCE. The only cut that may leave its piece's span is the
          // piece's own END, and the branch above has already answered every
          // one of those BY IDENTITY. So a cut arriving here is a statement
          // the tables would lose, and it cannot exist: measured 0 over the
          // t=0, 1e-6 and 1e-9 corpora, both domain regressions, the whole
          // suite and the 1e-4 scenes that drive the wave hardest, while the
          // endpoint answer above fires 1 to 129 times in the same runs. The
          // census keeps counting it so a release build still states the
          // loss it would otherwise take in silence.
          assert(proposal.parameter > Param(0) && proposal.parameter < whole);
          if (proposal.parameter <= Param(0) || proposal.parameter >= whole) {
            verdict[i] = verdict_out_of_span;
            continue;
          }
          const bool mirrored =
              parameters_in_name_frame &&
              plane_recovery_flat_edge_inverted(plane_recovery_flat_edge_ends(
                  def, n_flat, vertex_offsets));
          cuts[i] = {proposal.tier, proposal.root,
                     mirrored ? Param(whole - proposal.parameter)
                              : proposal.parameter,
                     id};
          verdict[i] = verdict_kept;
        }
      },
      tf::checked);
  std::size_t n_kept = 0;
  for (std::size_t i = 0; i < n_proposed; ++i) {
    const auto stated = verdict[i];
    if (stated == verdict_kept) {
      if (n_kept != i)
        cuts[n_kept] = cuts[i];
      ++n_kept;
      continue;
    }
    if (stated == verdict_on_endpoint)
      ++on_endpoint;
    else if (stated == verdict_out_of_span)
      ++out_of_span;
  }
  cuts.erase_till_end(cuts.begin() + std::ptrdiff_t(n_kept));

  // an edge's whole name is (tier, id): the pair leads every ordering
  tbb::parallel_sort(cuts.begin(), cuts.end(),
                     [](const cut_t &x, const cut_t &y) {
                       return std::tie(x.tier, x.edge, x.id, x.parameter) <
                              std::tie(y.tier, y.edge, y.id, y.parameter);
                     });
  cuts.erase_till_end(
      std::unique(cuts.begin(), cuts.end(), [](const cut_t &x, const cut_t &y) {
        return x.tier == y.tier && x.edge == y.edge && x.id == y.id;
      }));
  tbb::parallel_sort(cuts.begin(), cuts.end(),
                     [](const cut_t &x, const cut_t &y) {
                       return std::tie(x.tier, x.edge, x.parameter, x.id) <
                              std::tie(y.tier, y.edge, y.parameter, y.id);
                     });

  // the cuts are already grouped by their edge, so the group starts are an
  // ordered generate and every other output is a straight write
  split_offsets.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(cuts.size()), split_offsets,
      [&cuts](std::size_t at, tf::buffer<Index> &out) {
        if (at == 0 || cuts[at - 1].tier != cuts[at].tier ||
            cuts[at - 1].edge != cuts[at].edge)
          out.push_back(Index(at));
      },
      tf::checked);
  const auto n_groups = split_offsets.size();
  split_offsets.push_back(Index(cuts.size()));
  split_edge.allocate(n_groups);
  split_tier.allocate(n_groups);
  tf::parallel_for_each(
      tf::make_sequence_range(n_groups),
      [&](std::size_t group) {
        const auto &first = cuts[std::size_t(split_offsets[group])];
        split_edge[group] = first.edge;
        split_tier[group] = first.tier;
      },
      tf::checked);
  split_data.allocate(cuts.size());
  tf::parallel_for_each(
      tf::make_sequence_range(cuts.size()),
      [&](std::size_t at) { split_data[at] = cuts[at].id; }, tf::checked);
}

template <typename Index>
auto plane_recovery_edge_key(
    const tf::intersect::graph::plane_edge_def<Index> &def)
    -> std::array<Index, 4> {
  return {Index(def.point_tag_0), def.point_0, Index(def.point_tag_1),
          def.point_1};
}

template <typename Index>
auto find_retained_plane_edge_origin(
    const tf::intersect::graph::plane_edge_def<Index> &def,
    const tf::buffer<plane_birth_edge_record<Index>> &origin_index) -> Index {
  const auto key = plane_recovery_edge_key(def);
  const auto at = std::lower_bound(
      origin_index.begin(), origin_index.end(), key,
      [](const plane_birth_edge_record<Index> &record,
         const std::array<Index, 4> &value) {
        return std::tie(record[0], record[1], record[2], record[3]) <
               std::tie(value[0], value[1], value[2], value[3]);
      });
  return at != origin_index.end() &&
                 std::tie((*at)[0], (*at)[1], (*at)[2], (*at)[3]) ==
                     std::tie(key[0], key[1], key[2], key[3])
             ? (*at)[4]
             : Index(-1);
}

/// Find an exact edge origin in the immutable prefix or retained PA suffix.
template <typename Index, typename Immutable>
auto find_plane_edge_origin(
    const Immutable &immutable,
    const tf::intersect::graph::plane_edge_def<Index> &def,
    Index immutable_extent,
    const tf::buffer<plane_birth_edge_record<Index>> &origin_index) -> Index {
  const auto key = plane_recovery_edge_key(def);
  Index lo = 0;
  Index hi = immutable_extent;
  while (lo < hi) {
    const auto mid = lo + (hi - lo) / Index(2);
    if (plane_recovery_edge_key(immutable.canon_group(mid)[0]) < key)
      lo = mid + Index(1);
    else
      hi = mid;
  }
  if (lo < immutable_extent &&
      plane_recovery_edge_key(immutable.canon_group(lo)[0]) == key)
    return lo;
  return find_retained_plane_edge_origin(def, origin_index);
}

/// Name a recovery crossing by the two exact current pieces it crosses.
/// Point identities make one piece fit in two integers, so the structural
/// name remains stable when canonical group numbers change between waves.
template <typename Index, typename Tables, typename VertexOffsets>
auto make_plane_recovery_crossing_name(const Tables &tables, Index group_a,
                                       Index group_b, Index n_flat,
                                       const VertexOffsets &vertex_offsets,
                                       Index kind, Index &root, Index &partner)
    -> tf::intersect::graph::plane_name<Index> {
  auto key_a = plane_recovery_flat_edge_key(tables.canon_group(group_a)[0],
                                            n_flat, vertex_offsets);
  auto key_b = plane_recovery_flat_edge_key(tables.canon_group(group_b)[0],
                                            n_flat, vertex_offsets);
  root = group_a;
  partner = group_b;
  if (key_b < key_a) {
    std::swap(key_a, key_b);
    std::swap(root, partner);
  }
  return {kind, key_a[0], key_a[1], key_b[0], key_b[1]};
}

} // namespace tf::arrangement
