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
#include "../../core/views/sequence_range.hpp"
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per exposed triangle, the `(tag, object)` a reverse-side
///        emission is attributed to.
///
/// A coincident stack emits from its survivor, and the survivor's two
/// sides can belong to different members: the forward side to the
/// survivor itself, the reverse side to a member wound the other way. A
/// stack deeper than a pair can hold SEVERAL such members, and only one
/// name reaches a reverse emission, so the reverse side takes the
/// minimal `(tag, object)` among them — the same minimal-tag rule that
/// elected the survivor for the forward side. Without an opposing member
/// the label stays the triangle's own: the survivor carries both sides.
///
/// The triples are the arrangement's own stack records — the same
/// relation that decided triangle liveness, so attribution and liveness
/// cannot disagree.
///
/// @pre `triples` is sorted by `(survivor, dead)`, as the arrangement
///      build leaves it, so a survivor's records are contiguous. Order
///      groups them; it does not decide which one wins.
template <typename Index, typename Arrangement>
auto make_reverse_side_labels(const Arrangement &arrangement)
    -> tf::buffer<std::array<Index, 2>> {
  auto descriptors = arrangement.global().exposed_descriptors();
  auto slots = arrangement.triangle_slots();
  const auto n_exposed = Index(arrangement.triangle_tags().size());
  tf::buffer<std::array<Index, 2>> labels;
  labels.allocate(std::size_t(n_exposed));
  tf::parallel_for_each(
      tf::make_sequence_range(n_exposed), [&](Index e) {
        const auto &d = descriptors[std::size_t(slots[e])];
        labels[std::size_t(e)] = {static_cast<Index>(d.tag), d.object};
      });
  auto triples = arrangement.coplanar_triples();
  for (std::size_t i = 0; i < triples.size();) {
    const auto survivor = triples[i].survivor;
    std::size_t j = i;
    std::array<Index, 2> reverse{Index(-1), Index(-1)};
    for (; j < triples.size() && triples[j].survivor == survivor; ++j) {
      if (!triples[j].opposing)
        continue;
      const auto &d = descriptors[std::size_t(slots[triples[j].dead])];
      const std::array<Index, 2> member{static_cast<Index>(d.tag), d.object};
      if (reverse[0] == Index(-1) || member < reverse)
        reverse = member;
    }
    if (reverse[0] != Index(-1))
      labels[std::size_t(survivor)] = reverse;
    i = j;
  }
  return labels;
}

} // namespace tf::csg::graph
