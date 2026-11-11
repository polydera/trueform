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
#include "../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../core/algorithm/sequenced_generate.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../../core/views/drop.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "./tet_face_status.hpp"
#include "./tet_segment_phase.hpp"
#include "./tet_segment_status.hpp"
#include "./withdraw_tet_subsegment.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/parallel_sort.h>

namespace tf::topology::cdt::dt3 {

/// Retire the segments no live face holds any more: the rims of the faces
/// `retired` names, gathered into `candidates`, whose count of live faces
/// running along them (`phase.rim_faces`) falls to zero; each such segment
/// takes the status `retired`, its chain is no constraint, and its rows leave
/// the rows ticketed at their lower sites (@ref withdraw_tet_subsegment), so
/// a lookup there walks the rows of segments still constraints. The counts
/// are a diff: the faces stated since the last call, `status` not calling
/// them retired, are counted in, and the retired faces counted before are
/// counted out, each through `held`, so a call costs the rims it touches.
/// Returns the segments retired.
template <typename Index, typename Coord, typename Status>
auto retire_tet_segments(tet_segment_phase<Index, Coord> &phase,
                         const Status &status, const tf::buffer<Index> &retired,
                         tf::buffer<Index> &candidates, tf::buffer<Index> &held)
    -> std::size_t {
  auto &count = phase.rim_faces;
  const std::size_t counted = phase.rim_counted;
  const std::size_t n_faces = phase.edges.size();
  if (count.size() < phase.segments.size()) {
    const std::size_t known = count.size();
    count.reallocate(phase.segments.size());
    tf::parallel_fill(tf::drop(count, known), Index(0));
  }
  const auto rim = [&phase](std::size_t f) {
    return phase.edges[std::size_t(phase.edge_block[f])];
  };
  // the sorted multiset `held` applied to the counts, each segment's run by
  // its first entry, so the writes are disjoint
  const auto apply = [&count, &held](Index sign) {
    tbb::parallel_sort(held.begin(), held.end());
    tf::parallel_for_each(
        tf::make_sequence_range(held.size()),
        [&count, &held, sign](std::size_t i) {
          if (i && held[i - 1] == held[i])
            return;
          std::size_t end = i;
          while (end < held.size() && held[end] == held[i])
            ++end;
          count[std::size_t(held[i])] += sign * Index(end - i);
        },
        tf::checked);
  };
  held.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(counted, n_faces), held,
      [&status, &rim](std::size_t f, auto &out) {
        if (status[f] == tet_face_status::retired)
          return;
        for (auto s : rim(f))
          out.push_back(Index(s));
      },
      tf::checked);
  apply(Index(1));
  held.clear();
  candidates.clear();
  for (auto f : retired) {
    for (auto s : rim(std::size_t(f)))
      candidates.push_back(Index(s));
    if (std::size_t(f) < counted)
      for (auto s : rim(std::size_t(f)))
        held.push_back(Index(s));
  }
  apply(Index(-1));
  phase.rim_counted = n_faces;
  tbb::parallel_sort(candidates.begin(), candidates.end());
  candidates.erase_till_end(std::unique(candidates.begin(), candidates.end()));
  std::size_t retiring = 0;
  auto &segment = phase.segment;
  for (auto s : candidates)
    if (!count[std::size_t(s)]) {
      if (segment.status[std::size_t(s)] != tet_segment_status::retired)
        for (Index k = segment.first[std::size_t(s)]; k != Index(-1);
             k = segment.subsegments[std::size_t(k)].next) {
          const auto &row = segment.subsegments[std::size_t(k)];
          withdraw_tet_subsegment(segment, std::min(row.a, row.b), k);
        }
      segment.status[std::size_t(s)] = tet_segment_status::retired;
      ++retiring;
    }
  if constexpr (tf::core::verification)
    for (std::size_t site = 0; site < segment.row_heads.size(); ++site)
      for (Index k = segment.row_heads[site]; k != Index(-1);
           k = segment.row_links[std::size_t(k)].next)
        tf::core::verify(
            segment.status[std::size_t(
                segment.subsegments[std::size_t(k)].segment)] !=
                tet_segment_status::retired,
            "a site's chain holds only the rows of segments still a "
            "constraint");
  if constexpr (tf::core::verification) {
    tf::buffer<Index> expected;
    expected.allocate_and_initialize(count.size(), Index(0));
    for (std::size_t f = 0; f < n_faces; ++f)
      if (status[f] != tet_face_status::retired)
        for (auto s : rim(f))
          ++expected[std::size_t(s)];
    for (std::size_t s = 0; s < count.size(); ++s)
      tf::core::verify(count[s] == expected[s],
                       "a segment's rim count is the live faces along it");
  }
  return retiring;
}

} // namespace tf::topology::cdt::dt3
