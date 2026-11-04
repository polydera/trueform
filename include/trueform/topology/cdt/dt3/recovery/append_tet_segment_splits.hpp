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
#include "../../../../core/none.hpp"
#include "../../../../core/views/mapped_range.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../tet_minted_site.hpp"
#include "../tet_owner_complex.hpp"
#include "./split_tet_subsegment.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include "./tet_subsegment_stands.hpp"
#include "./ticket_tet_subsegment.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <tbb/parallel_sort.h>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Name this round's fresh splits from the complex's own counter, in the
/// canonical order of their records, append them as one batch and apply
/// every live split into rows one prefix reserves. A batch holding a ring's
/// split inserts serially in the append's canonical order, and a ring's
/// split whose row stands when its turn comes is declined: it mints
/// nothing, its ring stays owed and its row is read again. A ring's split
/// that lands realizes the ring. A batch the index type
/// cannot name refuses its segments for capacity, and one that would pass
/// `site_budget` sites for the budget. A withdrawn site leaves
/// its row due again; a round that changes nothing refuses the segments it
/// withdrew from as wrapped. Returns false when the append refused,
/// publishing no cells.
template <typename DT, typename Index, typename Coord>
auto append_tet_segment_splits(DT &dt, Index round,
                               tet_segment_recovery<Index, Coord> &recovery)
    -> bool {
  auto &splits = recovery.splits;
  const auto segment_of = [&recovery](const auto &split) {
    return recovery.subsegments[std::size_t(split.subsegment)].segment;
  };
  const auto rank = [&recovery, &segment_of](const auto &split) {
    if (recovery.status[std::size_t(segment_of(split))] !=
        tet_segment_status::pending)
      return 2;
    return split.site == Index(-1) ? 0 : 1;
  };
  tbb::parallel_sort(splits.begin(), splits.end(),
                     [&rank, &segment_of](const auto &x, const auto &y) {
                       if (rank(x) != rank(y))
                         return rank(x) < rank(y);
                       if (segment_of(x) != segment_of(y))
                         return segment_of(x) < segment_of(y);
                       if (x.numerator != y.numerator)
                         return x.numerator < y.numerator;
                       return x.subsegment < y.subsegment;
                     });
  const auto at = [&splits, &rank](int value) {
    return std::size_t(std::partition_point(splits.begin(), splits.end(),
                                            [&rank, value](const auto &split) {
                                              return rank(split) < value;
                                            }) -
                       splits.begin());
  };
  std::size_t fresh = at(1);
  const std::size_t live = at(2);

  const auto maximum = std::size_t(std::numeric_limits<Index>::max());
  const auto name = dt.next_name();
  std::size_t skipped = 0;
  const bool named = fresh <= maximum - dt.n_sites() && name <= maximum &&
                     fresh <= maximum - name;
  const bool afforded = fresh <= recovery.site_budget &&
                        dt.n_sites() <= recovery.site_budget - fresh;
  if (!named || !afforded) {
    const auto refusal =
        named ? tet_segment_status::budget : tet_segment_status::capacity;
    for (std::size_t i = 0; i < fresh; ++i) {
      const auto segment = std::size_t(segment_of(splits[i]));
      recovery.status[segment] = refusal;
      recovery.rounds[segment] = round;
    }
    skipped = fresh;
    fresh = 0;
  }
  const auto minted = tf::make_mapped_range(
      tf::make_sequence_range(fresh), [&splits, name](std::size_t i) {
        return tet_minted_site<Index, Coord>{Index(name + i),
                                             splits[i].position};
      });
  const std::size_t base = dt.index_map().f().size();
  recovery.declined.allocate(fresh);
  std::fill(recovery.declined.begin(), recovery.declined.end(), char(0));
  const auto keep = [&recovery, &splits](const auto &owner, std::size_t i) {
    if (splits[i].ring == Index(-1))
      return true;
    const tf::none_t serial{};
    const tet_owner_complex<false, std::remove_reference_t<decltype(owner)>,
                            const tf::none_t>
        complex(owner, serial);
    recovery.declined[i] =
        tet_subsegment_stands(complex, splits[i].subsegment, recovery);
    return !recovery.declined[i];
  };
  const bool rings =
      std::any_of(splits.begin(), splits.begin() + std::ptrdiff_t(fresh),
                  [](const auto &split) { return split.ring != Index(-1); });
  if (fresh &&
      !(rings ? dt.append_sites(minted, keep) : dt.append_sites(minted))) {
    for (std::size_t s = 0; s < recovery.status.size(); ++s)
      if (recovery.status[s] == tet_segment_status::pending) {
        recovery.status[s] = tet_segment_status::capacity;
        recovery.rounds[s] = round;
      }
    return false;
  }

  const auto &map = dt.index_map();
  const auto &f = map.f();
  auto &targets = recovery.targets;
  targets.allocate(live);
  recovery.placed_from = recovery.placed.size();
  std::size_t next = recovery.subsegments.size();
  std::size_t withdrawn = 0;
  for (std::size_t i = 0; i < live; ++i) {
    if (i < skipped) {
      targets[i] = Index(-1);
      continue;
    }
    const auto &split = splits[i];
    if (i < skipped + fresh) {
      const Index slot = f[base + (i - skipped)];
      if (slot == DT::k_none) {
        targets[i] = Index(-1);
        recovery.revisits.push_back(split.subsegment);
        withdrawn += !recovery.declined[i - skipped];
        continue;
      }
      recovery.placed.push_back(
          {slot, recovery.subsegments[std::size_t(split.subsegment)].hint});
      if (split.ring != Index(-1))
        recovery.rings[std::size_t(split.ring)] = -1;
    }
    targets[i] = Index(next++);
  }
  recovery.subsegments.reallocate(next);
  recovery.proposed.reallocate(next);
  recovery.minted.reallocate(dt.n_sites() - recovery.first_minted);
  tf::parallel_for_each(
      tf::make_sequence_range(live),
      [&recovery, &splits, &targets, &f, &segment_of, base, skipped,
       fresh](std::size_t i) {
        if (targets[i] == Index(-1))
          return;
        const auto &split = splits[i];
        Index site = split.site;
        if (i < skipped + fresh) {
          site = f[base + (i - skipped)];
          recovery.minted[std::size_t(site) - recovery.first_minted] =
              segment_of(split);
        }
        split_tet_subsegment(recovery, split.subsegment, targets[i], site,
                             split.numerator);
      },
      tf::checked);
  for (std::size_t i = 0; i < live; ++i) {
    if (targets[i] == Index(-1))
      continue;
    const Index k = splits[i].subsegment, next = targets[i];
    const auto &tail = recovery.subsegments[std::size_t(next)];
    recovery.revisits.push_back(k);
    recovery.revisits.push_back(next);
    ticket_tet_subsegment(recovery, tail.a, k);
    ticket_tet_subsegment(recovery, tail.a, next);
    ticket_tet_subsegment(recovery, tail.b, next);
  }

  if (withdrawn && recovery.placed.size() == recovery.placed_from &&
      live == skipped + fresh)
    for (std::size_t i = skipped; i < skipped + fresh; ++i) {
      const auto segment = std::size_t(segment_of(splits[i]));
      recovery.status[segment] = tet_segment_status::wrapped;
      recovery.rounds[segment] = round;
    }
  return true;
}

} // namespace tf::topology::cdt::dt3
