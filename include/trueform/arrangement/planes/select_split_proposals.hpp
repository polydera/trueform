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
#include "../../exact/meta.hpp"
#include "tbb/parallel_sort.h"
#include <cstddef>

namespace tf::arrangement {

/// @brief Order `proposals` by (edge, position) and fill `accepted` with the
///        positions in that order that become splits.
///
/// A proposal at either end of its edge, one quantum from the previous
/// acceptance on that edge, or one quantum from a split some table already
/// carries is one split too many; `has_existing` answers the last question
/// at the caller's currency. `key_of` and `param_of` read the edge and the
/// position off a proposal.
template <typename Int, typename Proposal, typename KeyOf, typename ParamOf,
          typename HasExisting>
auto select_split_proposals(tf::buffer<Proposal> &proposals,
                            const KeyOf &key_of, const ParamOf &param_of,
                            const HasExisting &has_existing,
                            tf::buffer<std::size_t> &accepted) -> void {
  using param_t = typename tf::exact::meta<Int>::param_type;
  tbb::parallel_sort(proposals.begin(), proposals.end(),
                     [&](const Proposal &a, const Proposal &b) {
                       const auto &key_a = key_of(a);
                       const auto &key_b = key_of(b);
                       if (key_a != key_b)
                         return key_a < key_b;
                       return param_of(a) < param_of(b);
                     });
  accepted.clear();
  accepted.reserve(proposals.size());
  // Same-split tolerance as tf::arrangement::has_adjacent_split: one quantum of
  // the grid the transported parameter defines.
  const param_t maximum = param_t(1) << tf::exact::meta<Int>::split_grid_bits;
  param_t previous_parameter = param_t(0);
  bool has_previous = false;
  for (std::size_t index = 0; index < proposals.size(); ++index) {
    if (index != 0 && key_of(proposals[index]) != key_of(proposals[index - 1]))
      has_previous = false;
    const param_t parameter = param_of(proposals[index]);
    // One snap step is one ulp of the real the lattice came from, so a
    // position within half a step of an end IS that end. Materializing it
    // would put a second identity on a vertex that already exists — the
    // rejection the crossing producer states on its own endpoints.
    if (parameter <= param_t(0) || maximum <= parameter)
      continue;
    if (has_previous && parameter - previous_parameter <= param_t(1))
      continue;
    if (has_existing(proposals[index]))
      continue;
    accepted.push_back(index);
    previous_parameter = parameter;
    has_previous = true;
  }
}

} // namespace tf::arrangement
