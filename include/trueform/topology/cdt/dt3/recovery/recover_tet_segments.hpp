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
#include "../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../../../../exact/meta.hpp"
#include "../reclaim_tet_dead_rows.hpp"
#include "../record_tet_site_cells.hpp"
#include "../tet_execution_tuning.hpp"
#include "./append_tet_segment_splits.hpp"
#include "./prepare_tet_segment_splits.hpp"
#include "./prepare_tet_segments.hpp"
#include "./propose_tet_crossing_splits.hpp"
#include "./propose_tet_subsegment_split.hpp"
#include "./read_tet_subsegments.hpp"
#include "./regions/run_tet_region_round.hpp"
#include "./recover_tet_edge_by_flips.hpp"
#include "./settle_tet_split_sites.hpp"
#include "./state_dirty_tet_subsegments.hpp"
#include "./tet_crossing_pair.hpp"
#include "./tet_piece_segment.hpp"
#include "./tet_refusing_cut.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include "./tet_subsegment_query.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Recover protected segments, each two site slots of the published complex of
/// `dt`, as chains of its edges, by flips first and splits last: the rows due
/// at the start are read once together (@ref read_tet_subsegments), and every
/// missing row is tried by flips along its segment from either end (@ref
/// recover_tet_edge_by_flips) in rounds whose link level rises by one each
/// round, a round that leaves as many rows missing as the one before counting
/// as a stall and the third stall freeing the level for one last round; the
/// dead rows the flips leave are reclaimed between rows
/// (@ref reclaim_tet_dead_rows), so a round's memory is its live rows and
/// what it wrote since, whatever its level. A row whose walk meets another segment's piece crossing it is split at the
/// crossing on its own frame and the piece at the same site, one site both
/// chains share (@ref propose_tet_crossing_splits), at that walk; a crossing
/// its frame cannot place, and a protected facet the walk meets the row
/// piercing, is a crossing no flip removes: it is handed to `cut` there
/// (@ref tet_crossing_pair), which states its resolution and the rows it made
/// due into `recovery.due` and answers whether it cut; the rows due join the
/// rows missing, and a crossing it does not cut refuses the segment as `depth`,
/// tried no further, the pair handed to `cut` its one statement; the rows
/// still missing are tried with the full search along the segment while a
/// round recovers any; the rest split at the site the segment passes through
/// or at their midpoint (@ref propose_tet_subsegment_split), each split landed
/// and appended one at a time (@ref settle_tet_split_sites, @ref
/// append_tet_segment_splits), its halves joining the rows due, so a split is
/// followed by the flips that recover its halves before another is made.
/// A FLIP ROUND RUNS ITS REGIONS FIRST where the phase asked for them
/// (@ref run_tet_region_round): each region takes the rows whose first end
/// stands in it, in the queue's order, and a row the admission stopped is
/// left in the queue for the round's own serial pass, which is the frontier
/// and escalates as it always did — a stop is a deferral and never a
/// refusal. Each
/// segment is split along its own frame, at most `depth` bits deep, `depth` at
/// most one below the lattice's parameter bits, and deeper only where
/// `recovery.deepens`; a segment with no split left is refused for depth. The
/// chains are published on the deepest frame a segment reached. A segment that
/// is an edge is never flipped away. The rounds are serial in the rows' own
/// order and no flip draws at random, so the chains are the same at any worker
/// count. The segments of `segments` before `standing` are the ones
/// `recovery` holds, their chains standing; only the rows the statement
/// adds and the ones `recovery.revisits` names are due
/// (@ref prepare_tet_segments). Returns false only when an append refused,
/// which publishes no cells.
template <typename DT, typename Segments, typename Index, typename Coord,
          typename Cut = tet_refusing_cut,
          typename Partition = tet_standing_partition>
auto recover_tet_segments(DT &dt, const Segments &segments,
                          std::size_t standing, int depth,
                          tet_segment_recovery<Index, Coord> &recovery,
                          const Cut &cut = Cut{},
                          const Partition &partition = Partition{}) -> bool {
  using Int = typename DT::int_type;
  depth = std::min(depth, tf::exact::meta<Int>::param_bits - 1);
  prepare_tet_segments(dt, segments, standing, depth, recovery);
  auto &owner = dt.owner();
  record_tet_site_cells(owner);
  auto &rows = recovery.subsegments;
  auto &scratch = recovery.flips;
  scratch.touched.clear();
  scratch.arena.clear();
  scratch.collected.clear();
  scratch.level_limit = -1;
  scratch.auto_level = 1;
  scratch.star_limit = -1;
  const auto is_segment = [&recovery](Index p, Index q) {
    return tet_piece_segment(recovery, p, q) != Index(-1);
  };
  auto &missing = recovery.queue;
  auto &next = recovery.deferred;
  auto &queued = recovery.queued;
  missing.clear();
  queued.allocate_and_initialize(rows.size(), char(0));
  const auto enqueue = [&missing, &queued](Index k) {
    if (std::size_t(k) >= queued.size()) {
      const std::size_t old = queued.size();
      queued.reallocate(std::size_t(k) + 1);
      std::fill(queued.begin() + std::ptrdiff_t(old), queued.end(), char(0));
    }
    if (queued[std::size_t(k)])
      return;
    queued[std::size_t(k)] = 1;
    missing.push_back(k);
  };
  read_tet_subsegments(dt, recovery);
  for (std::size_t i = 0; i < recovery.dirty.size(); ++i)
    if (!recovery.queries[i].present)
      enqueue(recovery.dirty[i]);
  Index step = 0;
  bool published = true;
  tet_subsegment_query<Index> query{};
  const auto try_flips_on = [&owner, &rows, &is_segment](
                                tet_flip_scratch<Index> &scratch, Index k,
                                bool fullsearch, Index &on,
                                std::array<Index, 2> &crossed,
                                Index &pierced) {
    const auto &row = rows[std::size_t(k)];
    Index found = Index(-1);
    std::array<Index, 2> met{Index(-1), Index(-1)};
    Index hit = Index(-1);
    on = Index(-1);
    if (recover_tet_edge_by_flips(owner, scratch, row.a, row.b, false,
                                  is_segment, found, crossed, pierced))
      return true;
    on = found;
    if (recover_tet_edge_by_flips(owner, scratch, row.b, row.a, false,
                                  is_segment, found, met, hit))
      return true;
    on = on != Index(-1) ? on : found;
    crossed = crossed[0] != Index(-1) ? crossed : met;
    pierced = pierced != Index(-1) ? pierced : hit;
    if (!fullsearch)
      return false;
    if (recover_tet_edge_by_flips(owner, scratch, row.a, row.b, true,
                                  is_segment, found, met, hit))
      return true;
    on = on != Index(-1) ? on : found;
    crossed = crossed[0] != Index(-1) ? crossed : met;
    pierced = pierced != Index(-1) ? pierced : hit;
    return false;
  };
  const auto realize = [&] {
    for (const auto &proposed : recovery.statement) {
      recovery.splits.clear();
      recovery.splits.push_back(proposed);
      settle_tet_split_sites(dt, recovery);
      if (!append_tet_segment_splits(dt, step, recovery)) {
        published = false;
        return;
      }
      state_dirty_tet_subsegments(dt, recovery);
      for (auto due : recovery.dirty)
        enqueue(due);
    }
  };
  const auto flip_task = [&](tet_flip_scratch<Index> &own, Index at) {
    const Index k = missing[std::size_t(at)];
    if (recovery.status[std::size_t(rows[std::size_t(k)].segment)] !=
        tet_segment_status::pending)
      return true;
    Index on = Index(-1);
    std::array<Index, 2> crossed{Index(-1), Index(-1)};
    Index pierced = Index(-1);
    return try_flips_on(own, k, false, on, crossed, pierced);
  };
  const auto anchor_of = [&](std::size_t at) -> Index {
    const Index site = rows[std::size_t(missing[at])].a;
    return site >= 0 && std::size_t(site) < recovery.site_patches.size()
               ? recovery.site_patches[std::size_t(site)]
               : Index(-1);
  };
  const auto round = [&](bool fullsearch, bool split) {
    next.clear();
    if (!fullsearch && !split &&
        run_tet_region_round(owner, partition, recovery.site_patches,
                             missing.size(), tet_region_pool_rows,
                             anchor_of, scratch, recovery.round_work,
                             flip_task, recovery.region_left, scratch)) {
      recovery.region_tasks += missing.size();
      recovery.region_deferred += recovery.region_left.size();
      recovery.flip_recovered += missing.size() - recovery.region_left.size();
      auto &left = recovery.region_left;
      for (std::size_t i = 0; i < left.size(); ++i)
        left[i] = missing[std::size_t(left[i])];
      for (auto k : missing)
        queued[std::size_t(k)] = 0;
      missing.clear();
      for (auto k : left)
        missing.push_back(k);
      for (auto k : missing)
        queued[std::size_t(k)] = 1;
    }
    for (std::size_t head = 0; head < missing.size(); ++head) {
      reclaim_tet_dead_rows<true>(owner);
      const Index k = missing[head];
      queued[std::size_t(k)] = 0;
      auto &row = rows[std::size_t(k)];
      const auto segment = std::size_t(row.segment);
      if (recovery.status[segment] != tet_segment_status::pending)
        continue;
      Index on = Index(-1);
      std::array<Index, 2> crossed{Index(-1), Index(-1)};
      Index pierced = Index(-1);
      const std::size_t capped = scratch.walk_capped;
      if (try_flips_on(scratch, k, fullsearch, on, crossed, pierced)) {
        ++(fullsearch ? recovery.search_recovered : recovery.flip_recovered);
        continue;
      }
      // a walk that spent its turns is as long as it is: the row refuses,
      // where a deferral and a split are both answers to another question
      if (scratch.walk_capped != capped) {
        recovery.status[segment] = tet_segment_status::passes;
        recovery.rounds[segment] = step;
        ++recovery.passes_capped;
        continue;
      }
      const Index across = crossed[0] != Index(-1)
                               ? tet_piece_segment(recovery, crossed[0], crossed[1])
                               : Index(-1);
      if (across != Index(-1) && across != row.segment) {
        recovery.splits.clear();
        if (propose_tet_crossing_splits<Int>(owner, recovery, k, crossed,
                                             step, recovery.splits)) {
          recovery.crossing_rows += 2;
          std::swap(recovery.splits, recovery.statement);
          realize();
          if (!published)
            return;
          ++step;
          continue;
        }
      }
      if ((across != Index(-1) && across != row.segment) ||
          pierced != Index(-1)) {
        recovery.due.rows.clear();
        recovery.due.faces.clear();
        if (cut(tet_crossing_pair<Index>{
                    row.segment, Index(-1),
                    {across != row.segment ? across : Index(-1), pierced}},
                recovery.due)) {
          for (auto due : recovery.due.rows)
            enqueue(due);
          continue;
        }
        recovery.status[segment] = tet_segment_status::depth;
        recovery.rounds[segment] = step;
        continue;
      }
      if (!split) {
        next.push_back(k);
        continue;
      }
      row.hint = owner._site_cells.load(std::size_t(row.a));
      query.on = on;
      query.hint = row.hint;
      query.encroacher = Index(-1);
      query.cosine = 0;
      query.present = false;
      query.encroached = false;
      recovery.splits.clear();
      recovery.refused.clear();
      propose_tet_subsegment_split<Int>(owner, recovery, k, query, step,
                                        recovery.splits, recovery.refused);
      for (auto refused : recovery.refused) {
        recovery.status[std::size_t(refused)] = tet_segment_status::depth;
        recovery.rounds[std::size_t(refused)] = step;
      }
      recovery.split_rows += recovery.splits.size();
      std::swap(recovery.splits, recovery.statement);
      realize();
      if (!published)
        return;
      ++step;
    }
    missing.clear();
    std::swap(missing, next);
    for (auto k : missing)
      queued[std::size_t(k)] = 1;
  };
  const auto fixpoint = [&] {
    std::size_t ms = missing.size();
    int stalls = 0;
    scratch.level_limit = -1;
    scratch.auto_level = 1;
    for (;;) {
      round(false, false);
      if (!published || missing.size() == 0)
        break;
      if (scratch.level_limit >= 0)
        break;
      if (missing.size() >= ms) {
        if (++stalls >= tet_execution_tuning::recovery_stall_rounds)
          scratch.level_limit = tet_execution_tuning::recovery_full_level;
      } else {
        ms = missing.size();
        if (stalls > 0)
          --stalls;
      }
      scratch.auto_level += 1;
    }
    while (published && missing.size()) {
      ms = missing.size();
      round(true, false);
      if (missing.size() >= ms)
        break;
    }
    if (published && missing.size()) {
      prepare_tet_segment_splits(dt, recovery);
      round(true, true);
    }
  };
  fixpoint();
  recovery.round += step;
  tf::parallel_for_each(
      tf::make_sequence_range(recovery.status.size()),
      [&recovery, step](std::size_t s) {
        if (recovery.status[s] == tet_segment_status::pending) {
          recovery.status[s] = tet_segment_status::recovered;
          recovery.rounds[s] = step;
        }
      },
      tf::checked);
  return published;
}

} // namespace tf::topology::cdt::dt3
