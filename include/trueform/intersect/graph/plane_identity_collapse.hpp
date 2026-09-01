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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/take.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_identity_names.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::intersect::graph {

/// The target a rewrite row states for `{source, id}`, or `-1` when the
/// identity was not moved. The table is closed, so one binary search is
/// the whole answer.
template <typename Index, typename Merges>
auto plane_merge_target(const Merges &merges, Index source, Index id) -> Index {
  const std::array<Index, 3> probe{source, id, Index(-1)};
  const auto at = std::lower_bound(
      merges.begin(), merges.end(), probe,
      [](const std::array<Index, 3> &x, const std::array<Index, 3> &y) {
        return std::tie(x[0], x[1]) < std::tie(y[0], y[1]);
      });
  if (at != merges.end() && (*at)[0] == source && (*at)[1] == id)
    return (*at)[2];
  return Index(-1);
}

/// The ORIGINALS a closed rewrite table retired, ascending and stated
/// once each: the table is sorted source-major and an original's source
/// is zero, so they are its leading run.
template <typename Index>
auto plane_retired_originals(const tf::buffer<std::array<Index, 3>> &merges) {
  std::size_t end = 0;
  while (end < merges.size() && merges[end][0] == Index(0))
    ++end;
  return tf::make_mapped_range(
      tf::make_range(merges.begin(), merges.begin() + std::ptrdiff_t(end)),
      [](const std::array<Index, 3> &row) { return row[1]; });
}

/// Close the rewrite rows into a flat table: a source stated with
/// several targets keeps the smallest and states the targets' own
/// identity, then every target chases to its root, so no row is a chain
/// and one binary search answers.
template <typename Index>
auto close_plane_merges(tf::buffer<std::array<Index, 3>> &merges) -> void {
  if (merges.size() == 0)
    return;
  tbb::parallel_sort(merges.begin(), merges.end());
  merges.erase_till_end(std::unique(merges.begin(), merges.end()));
  tf::buffer<std::array<Index, 3>> extra;
  std::size_t write = 0;
  for (std::size_t i = 0; i < merges.size();) {
    std::size_t j = i + 1;
    while (j < merges.size() && merges[j][0] == merges[i][0] &&
           merges[j][1] == merges[i][1])
      ++j;
    for (std::size_t k = i + 1; k < j; ++k)
      extra.push_back({Index(1), merges[k][2], merges[i][2]});
    merges[write++] = merges[i];
    i = j;
  }
  merges.erase_till_end(merges.begin() + std::ptrdiff_t(write));
  for (const auto &row : extra)
    merges.push_back(row);
  tbb::parallel_sort(merges.begin(), merges.end());
  merges.erase_till_end(std::unique(merges.begin(), merges.end()));
  for (auto &row : merges) {
    auto root = row[2];
    std::size_t guard = 0;
    for (auto next = plane_merge_target(merges, Index(1), root);
         next != Index(-1) && next != root && guard < merges.size();
         next = plane_merge_target(merges, Index(1), root), ++guard)
      root = next;
    row[2] = root;
  }
}

/// An endpoint as the closed table states it: a moved endpoint speaks
/// the created identity that absorbed it, so a consumer holding the old
/// `{tag, id}` stays merge-blind.
///
/// The table names an original by its FLAT vertex — the one key space
/// both producers of a rewrite row write — so the lift happens here and a
/// definition states its endpoint in its own per-mesh currency.
template <typename Index, typename Merges, typename VertexOffsets>
auto plane_merged_endpoint(const Merges &merges,
                           const VertexOffsets &vertex_offsets,
                           std::int16_t tag, Index id, std::int16_t &out_tag,
                           Index &out_id) -> void {
  const auto target =
      plane_merge_target(merges, tag < 0 ? Index(1) : Index(0),
                         tag < 0 ? id : vertex_offsets[std::size_t(tag)] + id);
  if (target == Index(-1)) {
    out_tag = tag;
    out_id = id;
    return;
  }
  out_tag = -1;
  out_id = target;
}

/// One definition's two endpoints as the closed table states them, in the
/// definition's own key order.
template <typename Index> struct plane_merged_endpoint_pair {
  std::int16_t lo_tag;
  Index lo_id;
  std::int16_t hi_tag;
  Index hi_id;
};

/// Both endpoints of one definition, read through
/// @ref tf::intersect::graph::plane_merged_endpoint.
template <typename Index, typename Merges, typename VertexOffsets>
auto plane_merged_endpoints(const Merges &merges,
                            const VertexOffsets &vertex_offsets,
                            const plane_edge_def<Index> &def)
    -> plane_merged_endpoint_pair<Index> {
  plane_merged_endpoint_pair<Index> ends{};
  plane_merged_endpoint(merges, vertex_offsets, def.point_tag_0, def.point_0,
                        ends.lo_tag, ends.lo_id);
  plane_merged_endpoint(merges, vertex_offsets, def.point_tag_1, def.point_1,
                        ends.hi_tag, ends.hi_id);
  return ends;
}

/// THE GATE — one producer of coordinates. Every class quantized once in
/// its own frame; then the collapse — identities that occupy one lattice
/// position are one identity, whatever produced them — and one dense
/// remap. THE GATE IS SHOWN EVERY IDENTITY THE CUT WORLD NAMES: both
/// endpoints of every definition enter the same table, so two originals
/// the lattice cannot tell apart are one identity whether or not any
/// split states them, and a split that rounds onto its root's endpoint is
/// absorbed by it rather than leaving a zero-length piece.
///
/// The table's rows are IDENTITIES, never mentions: a class states its
/// own, an endpoint states one only when nothing already does. A run of
/// one position therefore holds distinct identities by construction, and
/// a self weld cannot be stated. Sameness is BORN at this quantization,
/// which is why the discovery cannot move upstream — an exact producer
/// cannot see it.
///
/// `placed_originals` says a band placed the whole input. A placement is
/// what can put two originals on one lattice point, and it reaches
/// originals no cut face names, so under one EVERY original enters the
/// table; without one the mark set is what the definitions and the
/// landings state.
///
/// `landings` are the pairs the ROUNDING made — a created point and the
/// carrier endpoint it materialized onto — which no root's split states,
/// because no split was needed to create them. They enter as the two
/// identities they are, so the same election answers them.
///
/// `class_id` arrives holding the identity a class NAMES (`-1` when it
/// mints one) and leaves holding the identity it speaks; `created_class`
/// grows by one entry per minted identity; `merges` receives the closed
/// rewrite table. A position that holds only original vertices mints
/// too, and its class NAMES the smallest of them, so `statements` and
/// `class_offsets` grow by that class. The return value is the count of
/// positions that absorbed more than one identity.
template <typename Index, typename Graph, typename GetPoint,
          typename ClassPosition, typename VertexOffsets, typename Landings>
auto collapse_plane_identities(
    const Graph &g, const GetPoint &get_point,
    const ClassPosition &class_position, const VertexOffsets &vertex_offsets,
    const Landings &landings, bool placed_originals,
    tf::buffer<plane_name_statement<Index>> &statements,
    tf::buffer<Index> &class_offsets, Index n_classes, Index n_points,
    tf::buffer<Index> &class_id, tf::buffer<Index> &created_class,
    tf::buffer<std::array<Index, 3>> &merges) -> std::size_t {
  const auto defs = g.edge_defs();
  if (n_classes == 0 && defs.size() == 0 && landings.size() == 0)
    return 0;
  using pt3_t = std::decay_t<decltype(get_point(std::int16_t(-1), Index(0)))>;
  // the collapse's whole currency: a quantized position, the priority of
  // what states it (created identity, fresh class, original vertex) and
  // the identity itself
  struct position_row_t {
    pt3_t point;
    Index rank;
    Index id;
  };
  // THE ONE KEY SPACE both kinds of endpoint already live in: an original
  // by its flat vertex, a created point past the whole original space.
  const auto n_flat = vertex_offsets[vertex_offsets.size() - 1];
  const auto slot_of = [&vertex_offsets, n_flat](std::int16_t tag,
                                                 Index id) -> std::size_t {
    return std::size_t(tag < 0 ? n_flat + id
                               : vertex_offsets[std::size_t(tag)] + id);
  };
  const auto slot_point = [&](std::size_t slot) -> pt3_t {
    if (slot >= std::size_t(n_flat))
      return get_point(std::int16_t(-1), Index(slot) - n_flat);
    const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, Index(slot));
    return get_point(std::int16_t(tag),
                     Index(slot) - vertex_offsets[std::size_t(tag)]);
  };

  tf::buffer<position_row_t> positions;
  // the created identities a class already states, so an endpoint
  // repeating one is that class's row and not a second one
  tf::buffer<Index> named;
  positions.reserve(std::size_t(n_classes));
  named.reserve(std::size_t(n_classes));
  for (Index c = 0; c < n_classes; ++c) {
    const auto id = class_id[std::size_t(c)];
    positions.push_back(
        {class_position(c), id == Index(-1) ? 1 : 0, id == Index(-1) ? c : id});
    if (id != Index(-1))
      named.push_back(id);
  }
  tbb::parallel_sort(named.begin(), named.end());

  // ONE ROW PER IDENTITY, never per mention: an identity is named by as
  // many definitions and landings as reach it, and the key space is dense
  // and bounded, so a mark states it once and the gather reads it once. A
  // position group therefore holds distinct identities only, and a self
  // weld cannot be stated.
  tf::buffer<char> stated;
  stated.allocate(std::size_t(n_flat) + std::size_t(n_points));
  tf::parallel_fill(tf::take(stated, std::size_t(n_flat)),
                    placed_originals ? char(1) : char(0));
  tf::parallel_fill(tf::drop(stated, std::size_t(n_flat)), char(0));
  tf::parallel_for_each(
      defs,
      [&stated, &slot_of](const plane_edge_def<Index> &def) {
        stated[slot_of(def.point_tag_0, def.point_0)] = char(1);
        stated[slot_of(def.point_tag_1, def.point_1)] = char(1);
      },
      tf::checked);
  tf::parallel_for_each(
      landings,
      [&stated, &slot_of](const std::array<Index, 2> &landing) {
        stated[slot_of(std::int16_t(-1), landing[0])] = char(1);
        stated[std::size_t(landing[1])] = char(1);
      },
      tf::checked);
  positions.reserve(
      positions.size() +
      tf::reduce(tf::make_mapped_range(
                     tf::make_range(stated),
                     [](char mark) { return std::size_t(mark != char(0)); }),
                 [](std::size_t x, std::size_t y) { return x + y; },
                 std::size_t(0), tf::checked));
  tf::sequenced_generate(
      tf::make_sequence_range(stated.size()), positions,
      [&](std::size_t slot, tf::buffer<position_row_t> &out) {
        if (stated[slot] == char(0))
          return;
        const auto created = slot >= std::size_t(n_flat);
        const auto id = created ? Index(slot) - n_flat : Index(slot);
        if (created && std::binary_search(named.begin(), named.end(), id))
          return;
        out.push_back({slot_point(slot), created ? Index(0) : Index(2), id});
      },
      tf::checked);
  tbb::parallel_sort(
      positions.begin(), positions.end(),
      [](const position_row_t &x, const position_row_t &y) {
        return std::tie(x.point[0], x.point[1], x.point[2], x.rank, x.id) <
               std::tie(y.point[0], y.point[1], y.point[2], y.rank, y.id);
      });

  tf::buffer<std::array<Index, 3>> mergers;
  // the collision participants and nothing else: a position stating
  // itself more than once closes transitively here, so the election is
  // the closed class's and no chain ever reaches the rewrite table
  tf::buffer<Index> participants;
  tf::buffer<Index> parent;
  tf::buffer<Index> winner;
  auto find = [&](Index x) {
    while (parent[std::size_t(x)] != x) {
      parent[std::size_t(x)] = parent[std::size_t(parent[std::size_t(x)])];
      x = parent[std::size_t(x)];
    }
    return x;
  };
  // a fresh identity is only minted for a class that survives, so the
  // created id space stays dense
  auto mint = [&](Index cls) {
    created_class.push_back(cls);
    return n_points + Index(created_class.size()) - Index(1);
  };
  // the identity a closed class speaks, stated by its smallest member:
  // an existing created identity, else one minted for the class that
  // states the position, else — only originals here — one minted for a
  // class that NAMES the smallest of them, which is where its position
  // is read from
  auto elect = [&](const position_row_t &row) -> Index {
    if (row.rank == Index(0))
      return row.id;
    if (row.rank == Index(1))
      return mint(row.id);
    if (class_offsets.size() == 0)
      class_offsets.push_back(Index(0));
    statements.push_back(
        {plane_vertex_name<Index>(row.id), Index(-1), Index(-1), Index(-1)});
    class_offsets.push_back(Index(statements.size()));
    return mint(Index(class_offsets.size()) - Index(2));
  };

  std::size_t collapses = 0;
  std::size_t begin = 0;
  while (begin < positions.size()) {
    std::size_t end = begin + 1;
    while (end < positions.size() &&
           positions[end].point == positions[begin].point)
      ++end;
    // a lone statement of a position: a class holding no identity yet is
    // born here, an original keeps its own
    if (end - begin == 1) {
      if (positions[begin].rank == Index(1))
        class_id[std::size_t(positions[begin].id)] = elect(positions[begin]);
      begin = end;
      continue;
    }
    ++collapses;
    const auto first = Index(participants.size());
    for (auto i = begin; i < end; ++i) {
      participants.push_back(Index(i));
      parent.push_back(Index(participants.size()) - Index(1));
      winner.push_back(Index(-1));
    }
    for (auto k = first + Index(1); k < Index(participants.size()); ++k) {
      const auto a = find(first);
      const auto b = find(k);
      if (a != b)
        parent[std::size_t(a > b ? a : b)] = a < b ? a : b;
    }
    for (auto k = first; k < Index(participants.size()); ++k) {
      const auto &row = positions[std::size_t(participants[std::size_t(k)])];
      const auto root = find(k);
      // the positions are sorted, so a class is elected by the first
      // statement of it the scan reaches
      if (winner[std::size_t(root)] == Index(-1))
        winner[std::size_t(root)] = elect(row);
      const auto target = winner[std::size_t(root)];
      if (row.rank == Index(1))
        class_id[std::size_t(row.id)] = target;
      else if (row.rank == Index(0)) {
        if (row.id != target)
          mergers.push_back({Index(1), row.id, target});
      } else
        mergers.push_back({Index(0), row.id, target});
    }
    begin = end;
  }
  // the class's own losing members, in name currency: a class that NAMES
  // an existing point states the rewrite when the gate moved it, and a
  // class that names an original vertex always states one — originals
  // never survive a coincidence
  for (Index cls = 0; cls < n_classes; ++cls) {
    const auto target = class_id[std::size_t(cls)];
    if (target == Index(-1))
      continue;
    const auto &name =
        statements[std::size_t(class_offsets[std::size_t(cls)])].name;
    if (name[0] == Index(0) && name[1] != target)
      mergers.push_back({Index(1), name[1], target});
    else if (name[0] == Index(1))
      mergers.push_back({Index(0), name[1], target});
  }
  if (mergers.size() != 0) {
    tbb::parallel_sort(mergers.begin(), mergers.end());
    mergers.erase_till_end(std::unique(mergers.begin(), mergers.end()));
    for (Index c = 0; c < n_classes; ++c) {
      const auto target =
          plane_merge_target(mergers, Index(1), class_id[std::size_t(c)]);
      if (target != Index(-1))
        class_id[std::size_t(c)] = target;
    }
  }
  merges = std::move(mergers);
  return collapses;
}

} // namespace tf::intersect::graph
