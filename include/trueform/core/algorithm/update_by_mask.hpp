/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../index_map.hpp"
#include "./parallel_apply.hpp"
#include "./remove_if_and_make_map.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Update index map by removing masked elements.
///
/// Removes elements where mask is false and updates the mapping.
///
/// @tparam Index The index type.
/// @tparam Range0 The mask range type (bool-like elements).
/// @param im Index map to update (modified in-place).
/// @param mask Boolean mask (true = keep, false = remove).
template <typename Index, typename Range0>
auto update_by_mask(tf::index_map_buffer<Index> &im, const Range0 &mask) {
  tf::buffer<Index> map;
  map.allocate(im.kept_ids().size());
  auto none = Index(mask.size());
  auto it = tf::remove_if_and_make_map(
      im.kept_ids(), [&](Index id) { return !mask[id]; }, map, none);
  if (it == im.kept_ids().end())
    return;
  im.kept_ids().erase_till_end(it);
  tf::parallel_apply(
      im.f(),
      [&](Index &id) {
        if (id != none)
          id = map[id];
      },
      tf::checked);
}
} // namespace tf
