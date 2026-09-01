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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_identity_collapse.hpp"
#include "./plane_identity_names.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <tuple>
#include <utility>

namespace tf::intersect::graph {

/// A piece states itself: endpoints in ITS key order, the parent's
/// emission order preserved, the whole-side claim dropped when the
/// parent was cut, and the radial bit flipped exactly when the piece's
/// key order runs against the parent's.
template <typename Index>
auto plane_piece_def(plane_edge_def<Index> &piece,
                     const std::array<Index, 2> &from,
                     const std::array<Index, 2> &to,
                     const plane_edge_def<Index> &parent, Index count) -> void {
  auto a = from, b = to;
  const bool flipped = b < a;
  if (flipped)
    std::swap(a, b);
  piece.point_tag_0 = std::int16_t(a[0]);
  piece.point_0 = a[1];
  piece.point_tag_1 = std::int16_t(b[0]);
  piece.point_1 = b[1];
  std::uint8_t flags = parent.flags;
  if (count > 1)
    flags = std::uint8_t(flags & ~plane_edge_whole_side_flag);
  if (flipped) {
    flags = std::uint8_t(flags ^ plane_edge_reversed_flag);
    if (flags & plane_edge_radial_flag)
      flags = std::uint8_t(flags ^ plane_edge_radial_reversed_flag);
  }
  piece.flags = flags;
}

/// The ordered surviving identities of every root a junction cut: the
/// positional comparison runs span-local, on the quantized positions,
/// whose order along the root IS the exact fraction's order. An identity
/// that landed on an endpoint or outside the span is not a split of it.
///
/// A class states a cut only when it states a root, so a caller may name
/// any prefix of the class table: a naming class and a statement past the
/// named prefix both retire, and neither reaches the roots.
template <typename Index, typename Int, typename Graph, typename GetPoint,
          typename PositionOf, typename VertexOffsets>
auto order_plane_splits(
    const Graph &g, const GetPoint &get_point, const PositionOf &position_of,
    const tf::buffer<plane_name_statement<Index>> &statements,
    const tf::buffer<Index> &class_offsets, const tf::buffer<Index> &class_id,
    const tf::buffer<std::array<Index, 3>> &merges,
    const VertexOffsets &vertex_offsets, Index n_classes,
    tf::buffer<Index> &split_edge, tf::buffer<Index> &split_offsets,
    tf::buffer<Index> &split_data, std::size_t &out_of_span,
    std::size_t &on_endpoint) -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  // one statement of one junction, in the grain the roots are grouped by
  struct cut_t {
    Index edge;
    Index cls;
  };
  tf::buffer<cut_t> cuts;
  cuts.allocate(statements.size());
  // A statement that states no cut retires by sort key rather than by a
  // compacting walk: no canonical edge can be named `max`, so every slot
  // the classes do not cover, and every class that names a position
  // instead of splitting a root, lands past the live records in one
  // contiguous tail the group walk never reaches.
  const auto retired_edge = std::numeric_limits<Index>::max();
  tf::parallel_fill(cuts, cut_t{retired_edge, Index(-1)});
  tf::parallel_for_each(
      tf::make_sequence_range(n_classes),
      [&](Index cls) {
        const auto begin = std::size_t(class_offsets[std::size_t(cls)]);
        const auto end = std::size_t(class_offsets[std::size_t(cls) + 1]);
        for (auto i = begin; i < end; ++i)
          if (statements[i].root != Index(-1))
            cuts[i] = {statements[i].root, cls};
      },
      tf::checked);
  tbb::parallel_sort(cuts.begin(), cuts.end(),
                     [](const cut_t &x, const cut_t &y) {
                       return std::tie(x.edge, x.cls) < std::tie(y.edge, y.cls);
                     });
  cuts.erase_till_end(std::lower_bound(
      cuts.begin(), cuts.end(), retired_edge,
      [](const cut_t &cut, Index edge) { return cut.edge < edge; }));
  tf::buffer<Index> cut_offsets;
  if (cuts.size() != 0)
    tf::compute_offsets(
        cuts, std::back_inserter(cut_offsets), Index(0),
        [](const cut_t &x, const cut_t &y) { return x.edge == y.edge; });

  const auto n_groups =
      cut_offsets.size() == 0 ? std::size_t(0) : cut_offsets.size() - 1;
  split_edge.allocate(n_groups);
  for (std::size_t i = 0; i < n_groups; ++i)
    split_edge[i] = cuts[std::size_t(cut_offsets[i])].edge;

  split_data.clear();
  split_offsets.allocate(n_groups + 1);
  split_offsets[0] = 0;
  std::size_t offset_at = 1;
  struct keyed_id {
    T2 key;
    Index id;
  };
  struct local_t {
    tf::buffer<Index> sizes;
    tf::buffer<Index> data;
    tf::buffer<keyed_id> ordered;
    std::size_t out_of_span = 0;
    std::size_t on_endpoint = 0;
  };
  auto task = [&](auto &&range, local_t &local) {
    local.sizes.allocate(range.size());
    auto size_at = local.sizes.begin();
    for (const auto group : range) {
      const auto begin = std::size_t(cut_offsets[std::size_t(group)]);
      const auto end = std::size_t(cut_offsets[std::size_t(group) + 1]);
      const auto &def = g.canon_group(split_edge[std::size_t(group)])[0];
      const auto lo = get_point(def.point_tag_0, def.point_0);
      const auto hi = get_point(def.point_tag_1, def.point_1);
      const auto ends = plane_merged_endpoints(merges, vertex_offsets, def);
      const T1 d[3] = {T1(hi[0]) - lo[0], T1(hi[1]) - lo[1], T1(hi[2]) - lo[2]};
      const T2 span = T2(d[0]) * d[0] + T2(d[1]) * d[1] + T2(d[2]) * d[2];
      local.ordered.clear();
      for (auto i = begin; i < end; ++i) {
        const auto id = class_id[std::size_t(cuts[i].cls)];
        if (id == Index(-1))
          continue;
        if ((ends.lo_tag < 0 && id == ends.lo_id) ||
            (ends.hi_tag < 0 && id == ends.hi_id)) {
          ++local.on_endpoint;
          continue;
        }
        const auto point = position_of(id);
        const T2 key = T2(d[0]) * (T1(point[0]) - lo[0]) +
                       T2(d[1]) * (T1(point[1]) - lo[1]) +
                       T2(d[2]) * (T1(point[2]) - lo[2]);
        if (key <= T2(0) || key >= span) {
          ++local.out_of_span;
          continue;
        }
        local.ordered.push_back({key, id});
      }
      std::sort(local.ordered.begin(), local.ordered.end(),
                [](const keyed_id &x, const keyed_id &y) {
                  return x.key != y.key ? x.key < y.key : x.id < y.id;
                });
      Index previous = Index(-1);
      Index kept = 0;
      for (const auto &entry : local.ordered) {
        if (entry.id == previous)
          continue;
        previous = entry.id;
        local.data.push_back(entry.id);
        ++kept;
      }
      *size_at++ = kept;
    }
  };
  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.data, split_data);
    for (const auto size : local.sizes) {
      split_offsets[offset_at] = split_offsets[offset_at - 1] + size;
      ++offset_at;
    }
    out_of_span += local.out_of_span;
    on_endpoint += local.on_endpoint;
  };
  if (n_groups != 0)
    tf::blocked_reduce_sequenced_aggregate(
        tf::make_sequence_range(Index(n_groups)), tf::none, local_t{}, task,
        agg);
}

/// The canonical spans of the merged table, taken from the graph's own
/// spans instead of rediscovered: an untouched group keeps its span at a
/// shifted base, and the only boundaries the merge can create are the
/// piece groups — a few hundred records, so the whole structure costs
/// one walk of THEM, never of the table.
template <typename Index, typename Graph>
auto rebuild_plane_canon_spans(
    const Graph &g, const tf::buffer<plane_edge_def<Index>> &parts,
    const tf::buffer<Index> &insert, const tf::buffer<Index> &insert_group,
    const tf::buffer<Index> &untouched_base,
    const tf::buffer<Index> &untouched_rank, const tf::buffer<Index> &new_base,
    const tf::buffer<Index> &group_of_canon, Index untouched_groups,
    tf::buffer<plane_edge_def<Index>> &defs, tf::buffer<Index> &def_offsets)
    -> Index {
  def_offsets.clear();
  if (defs.size() == 0)
    return 0;
  // {first piece, the untouched group whose key it shares or -1}
  struct piece_group_t {
    Index at;
    Index joins;
  };
  tf::buffer<piece_group_t> piece_groups;
  for (std::size_t k = 0; k < parts.size();) {
    std::size_t e = k + 1;
    while (e < parts.size() && !plane_def_key_less(parts[k], parts[e]))
      ++e;
    const auto lo = insert_group[k];
    const bool coincident =
        lo < g.n_canon() && group_of_canon[std::size_t(lo)] == Index(-1) &&
        !plane_def_key_less(parts[k], g.canon_group(lo)[0]) &&
        !plane_def_key_less(g.canon_group(lo)[0], parts[k]);
    piece_groups.push_back({Index(k), coincident ? lo : Index(-1)});
    k = e;
  }
  // a fresh piece group is the only boundary the merge creates, so the
  // final index of anything is its own rank plus the fresh groups ahead
  // of it
  tf::buffer<Index> fresh_at;
  for (const auto &group : piece_groups)
    if (group.joins == Index(-1))
      fresh_at.push_back(insert[std::size_t(group.at)]);
  const auto n_canon = untouched_groups + Index(fresh_at.size());
  def_offsets.allocate(std::size_t(n_canon) + 1);
  def_offsets[std::size_t(n_canon)] = Index(defs.size());
  auto fresh_before = [&](Index position) {
    return Index(std::upper_bound(fresh_at.begin(), fresh_at.end(), position) -
                 fresh_at.begin());
  };
  tf::parallel_for_each(
      tf::make_sequence_range(g.n_canon()),
      [&](Index canon) {
        if (group_of_canon[std::size_t(canon)] != Index(-1))
          return;
        const auto base = untouched_base[std::size_t(canon)];
        def_offsets[std::size_t(untouched_rank[std::size_t(canon)] +
                                fresh_before(base))] =
            new_base[std::size_t(canon)];
      },
      tf::checked);
  Index fresh_rank = 0;
  for (const auto &group : piece_groups) {
    const auto start = insert[std::size_t(group.at)] + group.at;
    if (group.joins != Index(-1)) {
      def_offsets[std::size_t(
          untouched_rank[std::size_t(group.joins)] +
          fresh_before(untouched_base[std::size_t(group.joins)]))] = start;
      continue;
    }
    const auto lo = insert_group[std::size_t(group.at)];
    const auto ahead =
        lo < g.n_canon() ? untouched_rank[std::size_t(lo)] : untouched_groups;
    def_offsets[std::size_t(ahead + fresh_rank)] = start;
    ++fresh_rank;
  }
  tf::parallel_for_each(
      tf::make_sequence_range(n_canon),
      [&](Index group) {
        const auto begin = std::size_t(def_offsets[std::size_t(group)]);
        const auto end = std::size_t(def_offsets[std::size_t(group) + 1]);
        for (auto i = begin; i < end; ++i)
          defs[i].id = group;
      },
      tf::checked);
  return n_canon;
}

/// The plane CSR rebuilt off the OLD one: an untouched definition moved
/// by a monotone shift, so the block stays ordered by construction, and
/// only a plane holding a cut group needs its block ordered again.
template <typename Index, typename Graph>
auto rebuild_plane_edge_csr(const Graph &g, const tf::buffer<Index> &new_base,
                            const tf::buffer<Index> &pieces,
                            const tf::buffer<Index> &old_group_begin,
                            const tf::buffer<Index> &group_of_canon,
                            const tf::buffer<Index> &piece_base,
                            const tf::buffer<Index> &final_of_emission,
                            tf::offset_block_buffer<Index, Index> &edges)
    -> void {
  auto &offsets = edges.offsets_buffer();
  auto &data = edges.data_buffer();
  const auto n_planes = g.n_planes();
  offsets.allocate(std::size_t(n_planes) + 1);
  offsets[0] = 0;
  const auto old_defs = g.edge_defs();
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index p) {
        Index count = 0;
        for (const auto e : g.plane_edges(p))
          count += pieces[std::size_t(old_defs[std::size_t(e)].id)];
        offsets[std::size_t(p) + 1] = count;
      },
      tf::checked);
  for (std::size_t i = 1; i <= std::size_t(n_planes); ++i)
    offsets[i] += offsets[i - 1];
  data.allocate(std::size_t(offsets[std::size_t(n_planes)]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index p) {
        const auto begin = std::size_t(offsets[std::size_t(p)]);
        auto write = begin;
        bool cut = false;
        for (const auto e : g.plane_edges(p)) {
          const auto canon = old_defs[std::size_t(e)].id;
          const auto instance = Index(e) - old_group_begin[std::size_t(canon)];
          const auto group = group_of_canon[std::size_t(canon)];
          if (group == Index(-1)) {
            data[write++] = new_base[std::size_t(canon)] + instance;
            continue;
          }
          cut = true;
          const auto count = pieces[std::size_t(canon)];
          const auto base = std::size_t(piece_base[std::size_t(group)]) +
                            std::size_t(instance) * std::size_t(count);
          for (Index k = 0; k < count; ++k)
            data[write++] = final_of_emission[base + std::size_t(k)];
        }
        if (cut)
          std::sort(data.begin() + std::ptrdiff_t(begin),
                    data.begin() + std::ptrdiff_t(write));
      },
      tf::checked);
}

/// The entrant planes' blocks, appended to the plane CSR.
///
/// An entrant face is a plane of its own, so its definitions are one
/// block past the graph's planes, in the entrant table's own order — the
/// order a definition's stamp past the graph's descriptors states. The
/// blocks the graph's planes hold are not touched.
template <typename Index>
auto append_entrant_plane_edges(tf::buffer<std::array<Index, 2>> &rows,
                                tf::offset_block_buffer<Index, Index> &edges,
                                std::size_t n_blocks) -> void {
  if (n_blocks == 0)
    return;
  std::sort(rows.begin(), rows.end());
  auto &offsets = edges.offsets_buffer();
  auto &data = edges.data_buffer();
  const auto base = offsets.size() - 1;
  offsets.reallocate(offsets.size() + n_blocks);
  auto write = std::size_t(offsets[base]);
  data.reallocate(write + rows.size());
  std::size_t at = 0;
  for (std::size_t block = 0; block < n_blocks; ++block) {
    while (at < rows.size() && std::size_t(rows[at][0]) == block)
      data[write++] = rows[at++][1];
    offsets[base + 1 + block] = Index(write);
  }
}

template <typename Index>
auto append_entrant_plane_edges(tf::buffer<std::array<Index, 2>> &rows,
                                tf::offset_block_buffer<Index, Index> &edges)
    -> void {
  if (rows.size() == 0)
    return;
  Index last = 0;
  for (const auto &row : rows)
    last = row[0] > last ? row[0] : last;
  append_entrant_plane_edges(rows, edges, std::size_t(last) + 1);
}

/// A split found on one instance is a fact of the canonical group, so it
/// subdivides EVERY instance of that group — a contiguous span walk in
/// the canon-major table. The pieces ARE definitions: they inherit the
/// face, the original edge and the ordinal, and their radial bit is
/// their parent's, flipped exactly when the piece's own key order runs
/// against the parent's.
///
/// Only a touched group changes identity. Every other group's span is
/// already canonical and already in order, so the rebuild is a MERGE:
/// the untouched table keeps its order, the pieces are sorted once — a
/// few hundred records against a hundred thousand — and each one's
/// insertion point is a binary search over the canonical keys.
///
/// The entrant channels carry the definitions a face outside the cut
/// world states: `entrant_parts` are whole edges, which join a group of
/// their own key or found one, and claim no parent — a group they alone
/// hold sources `-1`; `entrant_instance_defs`, grouped by
/// `entrant_instance_offsets` over the split groups, are extra instances
/// of a cut group and ride its piece emission. A caller that supplies an
/// explicit entrant stamp base and extent retains empty carriers in their
/// aligned plane positions.
///
/// Both channels state their face as a stamp past the graph's
/// descriptors, contiguous in the order the entrant faces were stated,
/// and each of those faces is a plane of its own: their blocks are
/// appended to the CSR past the graph's planes, in that same order.
template <typename Index, typename Graph, typename VertexOffsets>
auto respan_plane_defs(
    const Graph &g, const tf::buffer<std::array<Index, 3>> &merges,
    const VertexOffsets &vertex_offsets, const tf::buffer<Index> &split_edge,
    const tf::buffer<Index> &split_offsets, const tf::buffer<Index> &split_data,
    const tf::buffer<plane_edge_def<Index>> &entrant_parts,
    const tf::buffer<Index> &entrant_instance_offsets,
    const tf::buffer<plane_edge_def<Index>> &entrant_instance_defs,
    tf::buffer<plane_edge_def<Index>> &defs, tf::buffer<Index> &def_offsets,
    tf::offset_block_buffer<Index, Index> &edges,
    Index entrant_stamp_base = Index(-1),
    Index entrant_plane_count = Index(-1)) -> Index {
  using def_t = plane_edge_def<Index>;
  const auto n_canon = g.n_canon();
  const auto n_groups = split_edge.size();

  tf::buffer<Index> pieces;
  tf::buffer<Index> group_of_canon;
  tf::buffer<Index> untouched_base;
  tf::buffer<Index> old_group_begin;
  pieces.allocate(std::size_t(n_canon));
  group_of_canon.allocate(std::size_t(n_canon));
  untouched_base.allocate(std::size_t(n_canon));
  old_group_begin.allocate(std::size_t(n_canon));
  tf::parallel_fill(pieces, Index(1));
  tf::parallel_fill(group_of_canon, Index(-1));
  for (std::size_t i = 0; i < n_groups; ++i) {
    pieces[std::size_t(split_edge[i])] =
        Index(split_offsets[i + 1] - split_offsets[i]) + 1;
    group_of_canon[std::size_t(split_edge[i])] = Index(i);
  }
  tf::buffer<Index> piece_base;
  tf::buffer<Index> untouched_rank;
  piece_base.allocate(n_groups + 1);
  untouched_rank.allocate(std::size_t(n_canon));
  Index untouched_at = 0, old_at = 0, rank_at = 0;
  for (Index canon = 0; canon < n_canon; ++canon) {
    const auto size = Index(g.canon_group(canon).size());
    old_group_begin[std::size_t(canon)] = old_at;
    old_at += size;
    untouched_base[std::size_t(canon)] = untouched_at;
    untouched_rank[std::size_t(canon)] = rank_at;
    if (group_of_canon[std::size_t(canon)] == Index(-1)) {
      untouched_at += size;
      ++rank_at;
    }
  }
  const bool has_entrant_instances = entrant_instance_offsets.size() != 0;
  Index piece_at = 0;
  for (std::size_t i = 0; i < n_groups; ++i) {
    piece_base[i] = piece_at;
    const auto extra = has_entrant_instances ? entrant_instance_offsets[i + 1] -
                                                   entrant_instance_offsets[i]
                                             : Index(0);
    piece_at += (Index(g.canon_group(split_edge[i]).size()) + extra) *
                pieces[std::size_t(split_edge[i])];
  }
  piece_base[n_groups] = piece_at;

  tf::buffer<def_t> parts;
  tf::buffer<Index> source_of_part;
  parts.allocate(std::size_t(piece_at) + entrant_parts.size());
  source_of_part.allocate(std::size_t(piece_at) + entrant_parts.size());
  tf::parallel_for_each(
      tf::make_sequence_range(Index(n_groups)),
      [&](Index group) {
        const auto canon = split_edge[std::size_t(group)];
        const auto span = g.canon_group(canon);
        const auto count = pieces[std::size_t(canon)];
        const auto splits = std::size_t(split_offsets[std::size_t(group)]);
        auto write = std::size_t(piece_base[std::size_t(group)]);
        auto emit = [&](const def_t &def) {
          const auto ends = plane_merged_endpoints(merges, vertex_offsets, def);
          std::array<Index, 2> current{Index(ends.lo_tag), ends.lo_id};
          for (Index k = 0; k < count; ++k) {
            const std::array<Index, 2> stop =
                k + 1 == count
                    ? std::array<Index, 2>{Index(ends.hi_tag), ends.hi_id}
                    : std::array<Index, 2>{Index(-1),
                                           split_data[splits + std::size_t(k)]};
            auto piece = def;
            plane_piece_def(piece, current, stop, def, count);
            piece.id = Index(write);
            source_of_part[write] = canon;
            parts[write++] = piece;
            current = stop;
          }
        };
        for (const auto &def : span)
          emit(def);
        if (has_entrant_instances) {
          const auto begin =
              std::size_t(entrant_instance_offsets[std::size_t(group)]);
          const auto end =
              std::size_t(entrant_instance_offsets[std::size_t(group) + 1]);
          for (auto i = begin; i < end; ++i)
            emit(entrant_instance_defs[i]);
        }
      },
      tf::checked);
  for (std::size_t i = 0; i < entrant_parts.size(); ++i) {
    const auto write = std::size_t(piece_at) + i;
    auto piece = entrant_parts[i];
    piece.id = Index(write);
    source_of_part[write] = Index(-1);
    parts[write] = piece;
  }
  tbb::parallel_sort(parts.begin(), parts.end(),
                     [](const def_t &x, const def_t &y) {
                       return plane_def_key_less(x, y) ||
                              (!plane_def_key_less(y, x) && x.id < y.id);
                     });

  tf::buffer<Index> insert;
  tf::buffer<Index> insert_group;
  insert.allocate(parts.size());
  insert_group.allocate(parts.size());
  const auto old_defs = g.edge_defs();
  tf::parallel_for_each(
      tf::make_sequence_range(Index(parts.size())),
      [&](Index k) {
        Index lo = 0, hi = n_canon;
        while (lo < hi) {
          const Index mid = lo + (hi - lo) / 2;
          if (plane_def_key_less(
                  old_defs[std::size_t(old_group_begin[std::size_t(mid)])],
                  parts[std::size_t(k)]))
            lo = mid + 1;
          else
            hi = mid;
        }
        insert_group[std::size_t(k)] = lo;
        insert[std::size_t(k)] =
            lo == n_canon ? untouched_at : untouched_base[std::size_t(lo)];
      },
      tf::checked);

  defs.allocate(std::size_t(untouched_at) + parts.size());
  tf::buffer<Index> new_base;
  new_base.allocate(std::size_t(n_canon));
  tf::parallel_for_each(
      tf::make_sequence_range(n_canon),
      [&](Index canon) {
        if (group_of_canon[std::size_t(canon)] != Index(-1)) {
          new_base[std::size_t(canon)] = Index(-1);
          return;
        }
        const auto base = untouched_base[std::size_t(canon)];
        const auto shift =
            Index(std::upper_bound(insert.begin(), insert.end(), base) -
                  insert.begin());
        new_base[std::size_t(canon)] = base + shift;
        auto write = std::size_t(base + shift);
        for (const auto &def : g.canon_group(canon))
          defs[write++] = def;
      },
      tf::checked);
  tf::buffer<Index> final_of_emission;
  final_of_emission.allocate(parts.size());
  tf::parallel_for_each(
      tf::make_sequence_range(Index(parts.size())),
      [&](Index k) {
        const auto at = insert[std::size_t(k)] + k;
        final_of_emission[std::size_t(parts[std::size_t(k)].id)] = at;
        defs[std::size_t(at)] = parts[std::size_t(k)];
      },
      tf::checked);

  const auto rebuilt = rebuild_plane_canon_spans(
      g, parts, insert, insert_group, untouched_base, untouched_rank, new_base,
      group_of_canon, rank_at, defs, def_offsets);
  rebuild_plane_edge_csr(g, new_base, pieces, old_group_begin, group_of_canon,
                         piece_base, final_of_emission, edges);
  if (entrant_parts.size() != 0 || has_entrant_instances ||
      entrant_plane_count != Index(-1)) {
    // {entrant plane, final definition}: a whole part is one emission
    // slot of its own, an extra instance owns the slots its group's
    // piece emission gave it. LA's all-live channel infers its base from
    // the smallest named stamp; PA states the source-face base and full
    // extent explicitly, so a collapsed face keeps its empty block.
    tf::buffer<std::array<Index, 2>> entrant_rows;
    auto stamp_base = entrant_stamp_base;
    if (stamp_base == Index(-1)) {
      stamp_base = std::numeric_limits<Index>::max();
      for (const auto &def : entrant_parts)
        stamp_base = def.face < stamp_base ? def.face : stamp_base;
      for (const auto &def : entrant_instance_defs)
        stamp_base = def.face < stamp_base ? def.face : stamp_base;
    }
    for (std::size_t i = 0; i < entrant_parts.size(); ++i)
      entrant_rows.push_back({entrant_parts[i].face - stamp_base,
                              final_of_emission[std::size_t(piece_at) + i]});
    if (has_entrant_instances)
      for (std::size_t group = 0; group < n_groups; ++group) {
        const auto canon = split_edge[group];
        const auto count = pieces[std::size_t(canon)];
        auto slot = std::size_t(piece_base[group]) +
                    g.canon_group(canon).size() * std::size_t(count);
        for (auto i = std::size_t(entrant_instance_offsets[group]);
             i < std::size_t(entrant_instance_offsets[group + 1]); ++i) {
          for (Index k = 0; k < count; ++k)
            entrant_rows.push_back({entrant_instance_defs[i].face - stamp_base,
                                    final_of_emission[slot + std::size_t(k)]});
          slot += std::size_t(count);
        }
      }
    if (entrant_plane_count == Index(-1))
      append_entrant_plane_edges(entrant_rows, edges);
    else
      append_entrant_plane_edges(entrant_rows, edges,
                                 std::size_t(entrant_plane_count));
  }
  return rebuilt;
}

/// The merge table applied to EVERY definition, and the canonicalization
/// that follows it: after this pass one undirected FINAL key is one
/// canonical group, so the tables are canonical by construction and no
/// consumer resolves an endpoint through the merge table.
///
/// Groups fuse; instances never do. A definition keeps its face, its
/// original edge, its ordinal and its direction — the reversed and
/// radial bits follow the key order exactly as a piece's do — so a
/// fused span states the union of what each of its instances stated.
/// A definition whose endpoints became ONE identity is no 1-cell at all
/// and leaves with the plane block entries that named it.
///
/// `id` carries a definition's own position through the sort, which is
/// the currency the plane CSR speaks: the blocks are remapped through
/// it and ordered again, so every block stays canon-grouped.
template <typename Index, typename VertexOffsets>
auto fuse_plane_defs(const tf::buffer<std::array<Index, 3>> &merges,
                     const VertexOffsets &vertex_offsets,
                     tf::buffer<plane_edge_def<Index>> &defs,
                     tf::buffer<Index> &def_offsets,
                     tf::offset_block_buffer<Index, Index> &edges,
                     tf::buffer<Index> *final_group_of_old = nullptr) -> Index {
  using def_t = plane_edge_def<Index>;
  const auto n_defs = Index(defs.size());
  if (n_defs == 0) {
    if (final_group_of_old)
      final_group_of_old->clear();
    return 0;
  }
  const auto n_old_canon = Index(def_offsets.size()) - Index(1);
  tf::buffer<Index> old_group_of_position;
  if (final_group_of_old) {
    old_group_of_position.allocate(std::size_t(n_defs));
    final_group_of_old->allocate(std::size_t(n_old_canon));
    tf::parallel_fill(*final_group_of_old, Index(-1));
    tf::parallel_for_each(
        tf::make_sequence_range(n_old_canon),
        [&](Index group) {
          for (auto i = def_offsets[std::size_t(group)];
               i < def_offsets[std::size_t(group) + 1]; ++i)
            old_group_of_position[std::size_t(i)] = group;
        },
        tf::checked);
  }
  // no live key can name it, so a collapsed definition retires by sort
  // key into one contiguous tail instead of by a compacting walk
  const auto retired_tag = std::numeric_limits<std::int16_t>::max();
  tf::parallel_for_each(
      tf::make_sequence_range(n_defs),
      [&](Index k) {
        const auto parent = defs[std::size_t(k)];
        const auto ends =
            plane_merged_endpoints(merges, vertex_offsets, parent);
        auto &def = defs[std::size_t(k)];
        def.id = k;
        if (ends.lo_tag == ends.hi_tag && ends.lo_id == ends.hi_id) {
          def.point_tag_0 = retired_tag;
          return;
        }
        plane_piece_def(def, {Index(ends.lo_tag), ends.lo_id},
                        {Index(ends.hi_tag), ends.hi_id}, parent, Index(1));
      },
      tf::checked);
  tbb::parallel_sort(defs.begin(), defs.end(),
                     [](const def_t &x, const def_t &y) {
                       return plane_def_key_less(x, y) ||
                              (!plane_def_key_less(y, x) && x.id < y.id);
                     });
  tf::buffer<Index> final_of_position;
  final_of_position.allocate(std::size_t(n_defs));
  tf::parallel_fill(final_of_position, Index(-1));
  defs.erase_till_end(std::lower_bound(defs.begin(), defs.end(), retired_tag,
                                       [](const def_t &def, std::int16_t tag) {
                                         return def.point_tag_0 < tag;
                                       }));
  def_offsets.clear();
  if (defs.size() == 0) {
    auto &offsets = edges.offsets_buffer();
    std::fill(offsets.begin(), offsets.end(), Index(0));
    edges.data_buffer().clear();
    return 0;
  }
  def_offsets.reserve(defs.size() + 1);
  tf::compute_offsets(defs, std::back_inserter(def_offsets), Index(0),
                      [](const def_t &x, const def_t &y) {
                        return x.point_tag_0 == y.point_tag_0 &&
                               x.point_0 == y.point_0 &&
                               x.point_tag_1 == y.point_tag_1 &&
                               x.point_1 == y.point_1;
                      });
  const auto n_canon = Index(def_offsets.size()) - Index(1);
  tf::parallel_for_each(
      tf::make_sequence_range(n_canon),
      [&](Index group) {
        const auto begin = std::size_t(def_offsets[std::size_t(group)]);
        const auto end = std::size_t(def_offsets[std::size_t(group) + 1]);
        for (auto i = begin; i < end; ++i) {
          const auto old_position = defs[i].id;
          final_of_position[std::size_t(old_position)] = Index(i);
          if (final_group_of_old)
            (*final_group_of_old)[std::size_t(
                old_group_of_position[std::size_t(old_position)])] = group;
          defs[i].id = group;
        }
      },
      tf::checked);

  auto &offsets = edges.offsets_buffer();
  auto &data = edges.data_buffer();
  const auto n_blocks = Index(offsets.size()) - Index(1);
  tf::buffer<Index> block_offsets;
  block_offsets.allocate(std::size_t(n_blocks) + 1);
  block_offsets[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&](Index p) {
        Index count = 0;
        for (auto i = std::size_t(offsets[std::size_t(p)]);
             i < std::size_t(offsets[std::size_t(p) + 1]); ++i)
          count += final_of_position[std::size_t(data[i])] == Index(-1)
                       ? Index(0)
                       : Index(1);
        block_offsets[std::size_t(p) + 1] = count;
      },
      tf::checked);
  for (std::size_t i = 1; i <= std::size_t(n_blocks); ++i)
    block_offsets[i] += block_offsets[i - 1];
  tf::buffer<Index> block_data;
  block_data.allocate(std::size_t(block_offsets[std::size_t(n_blocks)]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&](Index p) {
        const auto begin = std::size_t(block_offsets[std::size_t(p)]);
        auto write = begin;
        for (auto i = std::size_t(offsets[std::size_t(p)]);
             i < std::size_t(offsets[std::size_t(p) + 1]); ++i) {
          const auto at = final_of_position[std::size_t(data[i])];
          if (at != Index(-1))
            block_data[write++] = at;
        }
        std::sort(block_data.begin() + std::ptrdiff_t(begin),
                  block_data.begin() + std::ptrdiff_t(write));
      },
      tf::checked);
  offsets = std::move(block_offsets);
  data = std::move(block_data);
  return n_canon;
}

} // namespace tf::intersect::graph
