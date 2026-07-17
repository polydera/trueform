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
#include "../../core/views/enumerate.hpp"
#include "../../cut/arrangement_graph.hpp"
#include "../../cut/face_cuts.hpp"
#include <array>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Per-loop `(tag, object)` labels for reverse-side emission.
///
/// A coincident stack emits from its survivor, but a reverse emission
/// is attributed to the stack's smallest member wound that way — the
/// one whose stored winding faces out of the reverse side's domain.
/// Without an opposing member the label stays the loop's own: the
/// survivor carries both sides. The pairs are the arrangement's own
/// dedup pairs — the same relation that decided loop liveness, so
/// attribution and liveness cannot disagree.
///
/// @pre `ag.coplanar_pairs()` is sorted by (smaller, larger), as the
///      arrangement build leaves it: a survivor's pairs are contiguous
///      and its first opposing partner is its smallest.
template <typename Index, typename Int>
auto make_reverse_side_labels(const tf::arrangement_graph<Index> &ag,
                              const tf::face_cuts<Index, Int> &fc)
    -> tf::buffer<std::array<Index, 2>> {
  auto descriptors = fc.descriptors();
  tf::buffer<std::array<Index, 2>> labels;
  labels.allocate(descriptors.size());
  tf::parallel_for_each(tf::enumerate(labels), [&descriptors](auto pair) {
    auto &&[li, r] = pair;
    const auto &d = descriptors[li];
    r = {static_cast<Index>(d.tag), d.object};
  });
  Index twinned = -1;
  for (const auto &p : ag.coplanar_pairs())
    if (p[2] && p[0] != twinned) {
      twinned = p[0];
      const auto &d = descriptors[p[1]];
      labels[p[0]] = {static_cast<Index>(d.tag), d.object};
    }
  return labels;
}

} // namespace tf::csg::graph
