/*
* Copyright (c) 2025 XLAB
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
#include "../index_map.hpp"
#include "./keep_by_mask_and_make_map.hpp"
#include "./parallel_for_each.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Update index map by keeping only masked elements.
///
/// Removes elements from the index map where the mask is false.
/// The mask is indexed by position in kept_ids (0 to kept_ids.size()-1),
/// not by the original element IDs.
///
/// After this operation:
/// - kept_ids contains only elements where mask[position] was true
/// - f() is updated so that entries pointing to removed elements
///   are marked with a sentinel value (kept_ids.size())
///
/// @tparam Index The index type.
/// @tparam Range0 The mask range type (bool-like elements).
/// @param im Index map to update (modified in-place).
/// @param mask Boolean mask indexed by position (true = keep, false = remove).
///             Must have size >= kept_ids.size().
template <typename Index, typename Range0>
auto update_by_mask(tf::index_map_buffer<Index> &im, const Range0 &mask) {
  tf::buffer<Index> map;
  map.allocate(im.kept_ids().size());
  auto none = Index(im.kept_ids().size());
  auto it = tf::keep_by_mask_and_make_map(im.kept_ids(), mask, map, none);
  if (it == im.kept_ids().end())
    return;
  im.kept_ids().erase_till_end(it);
  tf::parallel_for_each(
      im.f(),
      [&](Index &id) {
        if (id != none)
          id = map[id];
      },
      tf::checked);
}
} // namespace tf
