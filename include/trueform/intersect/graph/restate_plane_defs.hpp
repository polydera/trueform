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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./plane_def_respan.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_identity_collapse.hpp"
#include "./plane_tables.hpp"
#include "./rebuild_plane_blocks.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::intersect::graph {

/// Statements merged into standing tables, the way the entrance merges its
/// parts (@ref tf::intersect::graph::respan_plane_defs): an untouched group
/// keeps its span at a shifted base and its rows their block by a monotone
/// shift, and only the parts are sorted. A touched group's live instances
/// are re-emitted once each through the closed `merges`; an instance of a
/// dying face, or one whose ends the merges joined, has no image. `rows`
/// join the merge on the planes `plane_of_row` names, which may lie past the
/// old carrier space up to `n_planes`. `touched` is ascending.
/// `row_images` states per old row the row it now stands at, `-1` for none.
template <typename Index, typename Int, typename VertexOffsets,
          typename IsDying, typename PlaneOfRow>
auto restate_plane_defs(const plane_tables<Index, Int> &old,
                        const tf::buffer<std::array<Index, 3>> &merges,
                        const VertexOffsets &vertex_offsets,
                        const tf::buffer<Index> &touched,
                        const IsDying &is_dying,
                        const tf::buffer<plane_edge_def<Index>> &rows,
                        const PlaneOfRow &plane_of_row, Index n_planes,
                        plane_tables<Index, Int> &out,
                        tf::buffer<Index> &row_images) -> void {
  using def_t = plane_edge_def<Index>;
  const auto n_canon = old.n_canon();
  const auto old_defs = old.edge_defs();
  const auto &old_offsets = old.def_offsets();

  tf::buffer<Index> group_of_canon;
  group_of_canon.allocate(std::size_t(n_canon));
  tf::parallel_fill(group_of_canon, Index(-1));
  for (std::size_t i = 0; i < touched.size(); ++i)
    group_of_canon[std::size_t(touched[i])] = Index(i);
  tf::buffer<Index> untouched_base;
  tf::buffer<Index> untouched_rank;
  untouched_base.allocate(std::size_t(n_canon));
  untouched_rank.allocate(std::size_t(n_canon));
  Index untouched_at = 0, rank_at = 0;
  for (Index canon = 0; canon < n_canon; ++canon) {
    untouched_base[std::size_t(canon)] = untouched_at;
    untouched_rank[std::size_t(canon)] = rank_at;
    if (group_of_canon[std::size_t(canon)] == Index(-1)) {
      untouched_at += Index(old.canon_group(canon).size());
      ++rank_at;
    }
  }

  // {part, its source}: an old row's index, or `-(k + 1)` for statement `k`
  struct part_t {
    def_t def;
    Index source;
  };
  tf::buffer<part_t> parts;
  const auto emit = [&](const def_t &parent, Index source) {
    const auto ends = plane_merged_endpoints(merges, vertex_offsets, parent);
    if (ends.lo_tag == ends.hi_tag && ends.lo_id == ends.hi_id)
      return;
    auto def = parent;
    plane_piece_def(def, {Index(ends.lo_tag), ends.lo_id},
                    {Index(ends.hi_tag), ends.hi_id}, parent, Index(1));
    parts.push_back({def, source});
  };
  for (const auto canon : touched)
    for (auto row = old_offsets[std::size_t(canon)];
         row < old_offsets[std::size_t(canon) + 1]; ++row)
      if (!is_dying(old_defs[std::size_t(row)].face))
        emit(old_defs[std::size_t(row)], row);
  for (std::size_t k = 0; k < rows.size(); ++k)
    emit(rows[k], -Index(k) - Index(1));
  tbb::parallel_sort(parts.begin(), parts.end(),
                     [](const part_t &x, const part_t &y) {
                       if (plane_def_key_less(x.def, y.def))
                         return true;
                       if (plane_def_key_less(y.def, x.def))
                         return false;
                       return plane_def_instance_less(x.def, y.def);
                     });

  tf::buffer<def_t> sorted;
  tf::buffer<Index> insert;
  tf::buffer<Index> insert_group;
  sorted.allocate(parts.size());
  insert.allocate(parts.size());
  insert_group.allocate(parts.size());
  tf::parallel_for_each(
      tf::make_sequence_range(Index(parts.size())),
      [&](Index k) {
        sorted[std::size_t(k)] = parts[std::size_t(k)].def;
        Index lo = 0, hi = n_canon;
        while (lo < hi) {
          const Index mid = lo + (hi - lo) / 2;
          if (plane_def_key_less(old.canon_group(mid)[0],
                                 parts[std::size_t(k)].def))
            lo = mid + 1;
          else
            hi = mid;
        }
        insert_group[std::size_t(k)] = lo;
        insert[std::size_t(k)] =
            lo == n_canon ? untouched_at : untouched_base[std::size_t(lo)];
      },
      tf::checked);

  auto &defs = out.defs();
  defs.allocate(std::size_t(untouched_at) + parts.size());
  // where each old row and each statement now stands; -1 for no image
  tf::buffer<Index> final_of_statement;
  row_images.allocate(old_defs.size());
  final_of_statement.allocate(rows.size());
  tf::parallel_fill(row_images, Index(-1));
  tf::parallel_fill(final_of_statement, Index(-1));
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
        auto write = base + shift;
        for (auto row = old_offsets[std::size_t(canon)];
             row < old_offsets[std::size_t(canon) + 1]; ++row) {
          row_images[std::size_t(row)] = write;
          defs[std::size_t(write++)] = old_defs[std::size_t(row)];
        }
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(Index(parts.size())),
      [&](Index k) {
        const auto at = insert[std::size_t(k)] + k;
        defs[std::size_t(at)] = parts[std::size_t(k)].def;
        const auto source = parts[std::size_t(k)].source;
        if (source >= Index(0))
          row_images[std::size_t(source)] = at;
        else
          final_of_statement[std::size_t(-source - Index(1))] = at;
      },
      tf::checked);
  rebuild_plane_canon_spans(old, sorted, insert, insert_group, untouched_base,
                            untouched_rank, new_base, group_of_canon, rank_at,
                            defs, out.def_offsets());

  // the plane CSR off the old one: an untouched row moves by its group's
  // shift, a re-emitted one to its image, a statement joins its plane, and
  // only a block that took a part is ordered again
  tf::buffer<std::array<Index, 2>> joining;
  for (std::size_t k = 0; k < rows.size(); ++k)
    if (final_of_statement[k] != Index(-1))
      joining.push_back({plane_of_row(rows[k]), final_of_statement[k]});
  std::sort(joining.begin(), joining.end());
  const auto old_planes = old.n_planes();
  const auto joined = [&](Index p) {
    const auto begin = std::lower_bound(joining.begin(), joining.end(),
                                        std::array<Index, 2>{p, Index(-1)});
    auto end = begin;
    while (end != joining.end() && (*end)[0] == p)
      ++end;
    return std::make_pair(begin, end);
  };
  auto &data = out.edges().data_buffer();
  rebuild_plane_blocks(
      std::size_t(n_planes),
      [&](std::size_t block) {
        const auto p = Index(block);
        Index count = 0;
        if (p < old_planes)
          for (const auto row : old.plane_edges(p))
            count += row_images[std::size_t(row)] == Index(-1) ? Index(0)
                                                               : Index(1);
        const auto range = joined(p);
        return count + Index(range.second - range.first);
      },
      [&](std::size_t block, std::size_t begin) {
        const auto p = Index(block);
        auto write = begin;
        bool moved = false;
        if (p < old_planes)
          for (const auto row : old.plane_edges(p)) {
            const auto at = row_images[std::size_t(row)];
            if (at == Index(-1)) {
              moved = true;
              continue;
            }
            moved =
                moved ||
                group_of_canon[std::size_t(old_defs[std::size_t(row)].id)] !=
                    Index(-1);
            data[write++] = at;
          }
        const auto range = joined(p);
        for (auto it = range.first; it != range.second; ++it)
          data[write++] = (*it)[1];
        if (moved || range.first != range.second)
          std::sort(data.begin() + std::ptrdiff_t(begin),
                    data.begin() + std::ptrdiff_t(write));
      },
      out.edges().offsets_buffer(), data, tf::checked);
}

} // namespace tf::intersect::graph
