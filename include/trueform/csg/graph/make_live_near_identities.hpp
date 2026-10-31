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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../intersect/graph/near_identity_rows.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>

namespace tf::csg::graph {

/// The near table of the arrangement `ag` — both tiers' — in the identities
/// its members speak now: each tier's deferred blocks joined at its window
/// (@ref tf::intersect::graph::join_near_identity_rows), each member read
/// through the composed layer, a pair that became one identity dropped, the
/// rest `{a, b}` with `a < b`, ascending and once, in the exposed stream's
/// flat ids. The one producer of "two standing identities within the
/// census's weld radius", and the only place a near table is joined.
template <typename ArrangementGraph>
auto make_live_near_identities(
    const ArrangementGraph &ag,
    tf::buffer<std::array<typename ArrangementGraph::index_type, 2>> &pairs)
    -> void {
  using Index = typename ArrangementGraph::index_type;
  pairs.clear();
  const auto &la_rows = ag.world().near_rows();
  const auto &pa_rows = ag.arrangement().near_rows();
  if (la_rows.n_blocks() + pa_rows.n_blocks() == 0)
    return;
  const auto &identities = ag.identities();
  const auto n_flat = ag.arrangement().n_flat_points();
  const auto live = [&identities, n_flat](Index flat) -> Index {
    const bool original = flat < n_flat;
    const auto target = identities.composed_target(
        original ? Index(0) : Index(1), original ? flat : flat - n_flat);
    return target == Index(-1) ? flat : n_flat + target;
  };
  const auto &class_ids = identities.local_class_ids();
  tf::intersect::graph::join_near_identity_rows(
      la_rows, ag.world().near_window(), n_flat,
      [&class_ids](Index cls) { return class_ids[std::size_t(cls)]; }, pairs);
  tf::intersect::graph::join_near_identity_rows(
      pa_rows, ag.arrangement().near_window(), n_flat,
      [](Index cls) { return cls; }, pairs);
  tf::parallel_for_each(
      pairs,
      [&live](std::array<Index, 2> &pair) {
        const auto a = live(pair[0]);
        const auto b = live(pair[1]);
        pair = a < b ? std::array<Index, 2>{a, b} : std::array<Index, 2>{b, a};
      },
      tf::checked);
  pairs.erase_till_end(std::remove_if(
      pairs.begin(), pairs.end(),
      [](const std::array<Index, 2> &pair) { return pair[0] == pair[1]; }));
  tbb::parallel_sort(pairs.begin(), pairs.end());
  pairs.erase_till_end(std::unique(pairs.begin(), pairs.end()));
}

} // namespace tf::csg::graph
