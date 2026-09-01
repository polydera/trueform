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

#include "../../core/buffer.hpp"
#include "./plane_round_evidence.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>

namespace tf::arrangement {

/// WAVE. The carriers of a round whose evidence is DISCARDED rather than
/// consumed: the planes that refused, and the planes a weld's substitution
/// reaches. Every other plane of that round stated nothing, so it keeps the
/// product it just built; these state their evidence again, against the
/// tables the barrier that discarded the round made real.
///
/// Both barriers ask this — the lazy world's first round and the wave
/// entrance — so the sentence has one producer.
template <typename Index, typename Int>
auto state_plane_round_frontier(const plane_round_evidence<Index, Int> &evidence,
                                tf::buffer<Index> &frontier) -> void {
  frontier.clear();
  frontier.reserve(evidence.refused.size() + evidence.welds.size());
  for (const auto plane : evidence.refused)
    frontier.push_back(plane);
  for (const auto &weld : evidence.welds)
    frontier.push_back(weld.plane);
  tbb::parallel_sort(frontier.begin(), frontier.end());
  frontier.erase_till_end(std::unique(frontier.begin(), frontier.end()));
}

} // namespace tf::arrangement
