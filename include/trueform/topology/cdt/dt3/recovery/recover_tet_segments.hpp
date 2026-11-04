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
#include "../../../../core/algorithm/reduce.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../../../../exact/meta.hpp"
#include "../record_tet_site_cells.hpp"
#include "./append_tet_segment_splits.hpp"
#include "./prepare_tet_segments.hpp"
#include "./propose_tet_subsegment_split.hpp"
#include "./publish_tet_segment_chains.hpp"
#include "./read_tet_subsegments.hpp"
#include "./recover_tet_edge_by_flips.hpp"
#include "./settle_tet_split_sites.hpp"
#include "./state_dirty_tet_subsegments.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include "./tet_subsegment_query.hpp"
#include <algorithm>
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
/// rows still missing are tried with the full search along the segment while a
/// round recovers any; the rest split at the site the segment passes through
/// or at their midpoint (@ref propose_tet_subsegment_split), each split landed
/// and appended one at a time (@ref settle_tet_split_sites, @ref
/// append_tet_segment_splits), its halves joining the rows due, so a split is
/// followed by the flips that recover its halves before another is made. Each
/// segment is split along its own frame, at most `depth` bits deep, `depth` at
/// most one below the lattice's parameter bits, and deeper only where
/// `recovery.deepens`; a segment with no split left is refused for depth. The
/// chains are published on the deepest frame a segment reached. A segment that
/// is an edge is never flipped away. The rounds are serial in the rows' own
/// order and no flip draws at random, so the chains are the same at any worker
/// count. Returns false only when an append refused, which publishes no cells.
template <typename DT, typename Segments, typename Index, typename Coord>
auto recover_tet_segments(DT &dt, const Segments &segments, int depth,
                          tet_segment_recovery<Index, Coord> &recovery)
    -> bool {
  using Int = typename DT::int_type;
  depth = std::min(depth, tf::exact::meta<Int>::param_bits - 1);
  prepare_tet_segments(dt, segments, depth, recovery);
  dt.record_killed_edges(true);
  auto &owner = dt.owner();
  record_tet_site_cells(owner);
  auto &rows = recovery.subsegments;
  auto &scratch = recovery.flips;
  scratch.counts.clear();
  scratch.arena.clear();
  scratch.collected.clear();
  scratch.encroached.clear();
  scratch.level_limit = -1;
  scratch.auto_level = 1;
  scratch.star_limit = -1;
  scratch.hull_barred = 0;
  recovery.flip_recovered = 0;
  recovery.search_recovered = 0;
  recovery.split_rows = 0;
  const auto is_segment = [&recovery](Index p, Index q) {
    const Index low = std::min(p, q), high = std::max(p, q);
    const auto &heads = recovery.row_heads;
    if (std::size_t(low) >= heads.size())
      return false;
    for (Index t = heads[std::size_t(low)]; t != Index(-1);
         t = recovery.row_tickets[std::size_t(t)].second) {
      const auto &row = recovery.subsegments[std::size_t(
          recovery.row_tickets[std::size_t(t)].first)];
      if ((row.a == low && row.b == high) || (row.a == high && row.b == low))
        return true;
    }
    return false;
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
  const auto try_flips = [&owner, &scratch, &rows, &is_segment](
                             Index k, bool fullsearch, Index &on) {
    const auto &row = rows[std::size_t(k)];
    Index found = Index(-1);
    on = Index(-1);
    if (recover_tet_edge_by_flips(owner, scratch, row.a, row.b, false,
                                  is_segment, found))
      return true;
    on = found;
    if (recover_tet_edge_by_flips(owner, scratch, row.b, row.a, false,
                                  is_segment, found))
      return true;
    on = on != Index(-1) ? on : found;
    if (!fullsearch)
      return false;
    if (recover_tet_edge_by_flips(owner, scratch, row.a, row.b, true,
                                  is_segment, found))
      return true;
    on = on != Index(-1) ? on : found;
    return false;
  };
  const auto round = [&](bool fullsearch, bool split) {
    next.clear();
    for (std::size_t head = 0; head < missing.size(); ++head) {
      const Index k = missing[head];
      queued[std::size_t(k)] = 0;
      auto &row = rows[std::size_t(k)];
      const auto segment = std::size_t(row.segment);
      if (recovery.status[segment] != tet_segment_status::pending)
        continue;
      Index on = Index(-1);
      if (try_flips(k, fullsearch, on)) {
        ++(fullsearch ? recovery.search_recovered : recovery.flip_recovered);
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
      ++step;
    }
    missing.clear();
    std::swap(missing, next);
    for (auto k : missing)
      queued[std::size_t(k)] = 1;
  };
  std::size_t ms = missing.size();
  int stalls = 0;
  for (;;) {
    round(false, false);
    if (!published || missing.size() == 0)
      break;
    if (scratch.level_limit >= 0)
      break;
    if (missing.size() >= ms) {
      if (++stalls >= 3)
        scratch.level_limit = 100000;
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
  if (published && missing.size())
    round(true, true);
  dt.record_killed_edges(false);
  recovery.round = step;
  tf::parallel_for_each(
      tf::make_sequence_range(recovery.status.size()),
      [&recovery, step](std::size_t s) {
        if (recovery.status[s] == tet_segment_status::pending) {
          recovery.status[s] = tet_segment_status::recovered;
          recovery.rounds[s] = step;
        }
      },
      tf::checked);
  publish_tet_segment_chains(tf::reduce(
                                 recovery.bits,
                                 [](int a, int b) { return std::max(a, b); },
                                 depth, tf::checked),
                             recovery);
  return published;
}

} // namespace tf::topology::cdt::dt3
