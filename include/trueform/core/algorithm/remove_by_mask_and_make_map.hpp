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
#include "./keep_by_mask_and_make_map.hpp"
#include "../views/mapped_range.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Remove elements by mask and create mapping to new indices.
///
/// Removes elements where mask is true while building a map from
/// old indices to new compact indices. Mask is indexed by position.
/// Forwards to keep_by_mask_and_make_map with negated mask.
///
/// @tparam Range0 The data range type.
/// @tparam Range1 The mask range type (bool-like, indexed by position).
/// @tparam Range2 The map range type.
/// @tparam Index The index type.
/// @param data Data to filter (modified in-place).
/// @param mask Boolean mask indexed by position (true = remove).
/// @param map Output map from old to new indices.
/// @param none_tag Value for removed element mappings.
/// @return Iterator to new end of data.
template <typename Range0, typename Range1, typename Range2, typename Index>
auto remove_by_mask_and_make_map(Range0 &data, const Range1 &mask, Range2 &map,
                                  Index none_tag) {
  return keep_by_mask_and_make_map(
      data, tf::make_mapped_range(mask, [](bool v) { return !v; }), map,
      none_tag);
}

/// @ingroup core_algorithms
/// @brief Remove elements by mask with default none tag (data.size()).
/// @overload
template <typename Range0, typename Range1, typename Range2>
auto remove_by_mask_and_make_map(Range0 &data, const Range1 &mask,
                                  Range2 &map) {
  using Index = std::decay_t<decltype(map[0])>;
  return remove_by_mask_and_make_map(data, mask, map, Index(data.size()));
}
} // namespace tf
