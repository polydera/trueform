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
#include "./point_delivery.hpp"
#include "./tagged_intersection.hpp"

#include <algorithm>
#include <cstddef>
#include <tuple>

namespace tf::intersect {

/// THE FACE CARRIER: every face a point reached, whichever currency
/// reached it.
///
/// A face is here because a pair record names it or because a delivery
/// does, and the two currencies take the SAME block positions — block
/// `i` is face `i`'s pairs and face `i`'s deliveries, either of which may
/// be empty. Position alone joins a face's loop to its pair groups, with
/// no map between them.
///
/// Both inputs arrive ordered by their own dedup, whose leading key is
/// `key()`, so one adjacent sweep of the two states the union.
template <typename Index>
auto group_intersection_carriers(
    const tf::buffer<tagged_intersection<Index>> &records,
    const tf::buffer<point_delivery<Index>> &deliveries, Index n_tags,
    tf::buffer<Index> &record_offsets, tf::buffer<Index> &delivery_offsets,
    tf::buffer<Index> &tag_offsets) -> void {
  record_offsets.clear();
  delivery_offsets.clear();
  tag_offsets.allocate(std::size_t(n_tags) + 1);
  std::fill(tag_offsets.begin(), tag_offsets.end(), Index(0));
  if (records.size() == 0 && deliveries.size() == 0)
    return;
  record_offsets.reserve(records.size() + deliveries.size() + 1);
  delivery_offsets.reserve(records.size() + deliveries.size() + 1);

  std::size_t r = 0, d = 0;
  record_offsets.push_back(Index(0));
  delivery_offsets.push_back(Index(0));
  while (r < records.size() || d < deliveries.size()) {
    const bool take_record =
        d == deliveries.size() ||
        (r != records.size() && records[r].key() < deliveries[d].key());
    const auto key = take_record ? records[r].key() : deliveries[d].key();
    while (r < records.size() && records[r].key() == key)
      ++r;
    while (d < deliveries.size() && deliveries[d].key() == key)
      ++d;
    record_offsets.push_back(Index(r));
    delivery_offsets.push_back(Index(d));
  }

  const auto n_blocks = Index(record_offsets.size() - 1);
  const auto block_tag = [&](Index b) {
    const auto at = std::size_t(b);
    return record_offsets[at] != record_offsets[at + 1]
               ? Index(records[std::size_t(record_offsets[at])].tag)
               : Index(deliveries[std::size_t(delivery_offsets[at])].tag);
  };
  tag_offsets[0] = Index(0);
  Index b = 0;
  for (Index t = 0; t < n_tags; ++t) {
    while (b < n_blocks && block_tag(b) == t)
      ++b;
    tag_offsets[std::size_t(t) + 1] = b;
  }
}

} // namespace tf::intersect
