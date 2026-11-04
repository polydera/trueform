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
#include "../../../../exact/meta.hpp"
#include "../tet_deep_lattice.hpp"
#include "../tet_deepest_of.hpp"
#include "../tet_position.hpp"
#include "../tet_site_position.hpp"
#include "../tet_with_sites.hpp"
#include "../tetrahedralization_owner.hpp"
#include "./deepen_tet_segment_frame.hpp"
#include "./tet_segment_numerator.hpp"
#include "./tet_segment_point.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_split_point.hpp"
#include "./tet_segment_status.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// State the split row `k` asks after reading `query`, once in `round`, into
/// `splits`, or its segment into `refused` for depth: nothing for a row whose
/// segment is settled or that is present; a row meeting a site splits at
/// that site; a row at a corner whose ring it still owes splits on the ring;
/// any other row with an encroacher splits where @ref tet_segment_split_point
/// asks, rounded to the nearest numerator strictly inside the row, and every
/// other at its exact midpoint, minted where it stands. A row with no
/// numerator inside deepens its segment's frame where `recovery.deepens`
/// (@ref deepen_tet_segment_frame); a split standing below the frame is
/// admitted only there and as deep as the owner's deep lattice admits
/// (@ref tet_deep_lattice), and a row with neither refuses.
template <typename Int, typename Index, typename Coord, typename Query,
          typename Splits, typename Refused>
auto propose_tet_subsegment_split(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    tet_segment_recovery<Index, Coord> &recovery, Index k, const Query &query,
    Index round, Splits &splits, Refused &refused) -> void {
  const auto &rows = recovery.subsegments;
  const auto &row = rows[std::size_t(k)];
  const auto segment = std::size_t(row.segment);
  if (recovery.status[segment] != tet_segment_status::pending ||
      query.present || recovery.proposed[std::size_t(k)] == round)
    return;
  recovery.proposed[std::size_t(k)] = round;
  const Index head = rows[std::size_t(recovery.first[segment])].a;
  const Index tail = rows[std::size_t(recovery.last[segment])].b;
  const bool framed = row.n0 >= 0 && row.n1 >= 0;
  const Index on = query.on;
  if (on != Index(-1)) {
    const int bits = recovery.bits[segment];
    const std::int64_t numerator =
        framed ? tet_with_sites(owner, tet_deepest_of(owner, head, tail, on),
                                [&](const auto &sites) {
                                  return tet_segment_numerator(sites, head,
                                                               tail, on, bits);
                                })
               : std::int64_t(-1);
    splits.push_back({k, numerator, tet_site_position(owner, on), on});
    return;
  }
  const auto minted = [&owner](Index from, Index to, std::int64_t numerator,
                               int bits) {
    return tet_with_sites(
        owner, tet_deepest_of(owner, from, to), [&](const auto &sites) {
          using lattice = typename std::decay_t<decltype(sites)>::int_type;
          return tet_minted_position<Coord>(
              tet_segment_point<typename tf::exact::meta<lattice>::T2>(
                  sites[std::size_t(from)].pt, sites[std::size_t(to)].pt,
                  numerator, bits),
              0, sites.depth + bits);
        });
  };
  const auto admitted = [&recovery](const tet_position<Coord> &position) {
    return !position.depth ||
           (recovery.deepens &&
            position.depth <= tet_deep_lattice<Int, Coord>::cap);
  };
  for (;;) {
    const int bits = recovery.bits[segment];
    if (framed && row.n1 - row.n0 >= 2) {
      std::int64_t numerator = row.n0 + (row.n1 - row.n0) / 2;
      Index ring = Index(-1);
      for (std::size_t end = 0; end < 2 && ring == Index(-1); ++end) {
        const std::int64_t owed = recovery.rings[2 * segment + end];
        if ((end ? recovery.last[segment] : recovery.first[segment]) == k &&
            owed > row.n0 && owed < row.n1) {
          numerator = owed;
          ring = Index(2 * segment + end);
        }
      }
      const Index encroacher = query.encroacher;
      const auto acute_end = [&recovery, k, segment](Index site) {
        if (recovery.first[segment] != k && recovery.last[segment] != k)
          return false;
        const auto &ends = recovery.ends;
        const auto at = std::lower_bound(
            ends.begin(), ends.end(), site,
            [](const auto &end, Index value) { return end.site < value; });
        return at != ends.end() && at->site == site &&
               bool(recovery.acute[std::size_t(at - ends.begin())]);
      };
      if (ring == Index(-1) && encroacher != Index(-1)) {
        const double t = tet_with_sites(
            owner, tet_deepest_of(owner, row.a, row.b, encroacher),
            [&](const auto &sites) {
              return tet_segment_split_point(
                  sites[std::size_t(row.a)].pt, sites[std::size_t(row.b)].pt,
                  sites[std::size_t(encroacher)].pt, acute_end(row.a),
                  acute_end(row.b));
            });
        if (t != 0.5)
          numerator = std::min(
              row.n1 - 1,
              std::max(row.n0 + 1, row.n0 + std::int64_t(std::llround(
                                                t * double(row.n1 - row.n0)))));
      }
      const auto position = minted(head, tail, numerator, bits);
      if (!admitted(position))
        break;
      splits.push_back({k, numerator, position, Index(-1), ring});
      return;
    }
    if (!framed && row.halvings < bits) {
      const auto position = minted(row.a, row.b, 1, 1);
      if (!admitted(position))
        break;
      splits.push_back({k, -1, position, Index(-1)});
      return;
    }
    if (!recovery.deepens ||
        bits + 1 >= std::min(tf::exact::meta<Int>::param_bits, 63))
      break;
    deepen_tet_segment_frame(recovery, Index(segment));
  }
  refused.push_back(row.segment);
}

} // namespace tf::topology::cdt::dt3
