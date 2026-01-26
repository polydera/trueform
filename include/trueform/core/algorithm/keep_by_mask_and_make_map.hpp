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
#include <type_traits>

namespace tf {

/// @ingroup core_algorithms
/// @brief Keep elements by mask and create mapping to new indices.
///
/// Keeps elements where mask is true while building a map from
/// old indices to new compact indices. Mask is indexed by position.
///
/// @tparam Range0 The data range type.
/// @tparam Range1 The mask range type (bool-like, indexed by position).
/// @tparam Range2 The map range type.
/// @tparam Index The index type.
/// @param data Data to filter (modified in-place).
/// @param mask Boolean mask indexed by position (true = keep).
/// @param map Output map from old to new indices.
/// @param none_tag Value for removed element mappings.
/// @return Iterator to new end of data.
template <typename Range0, typename Range1, typename Range2, typename Index>
auto keep_by_mask_and_make_map(Range0 &data, const Range1 &mask, Range2 &map,
                                Index none_tag) {
  Index current_id = 0;
  auto it0 = data.begin();
  auto write_to = it0;
  auto end0 = data.end();
  auto it1 = map.begin();
  auto it_mask = mask.begin();
  for (; it0 != end0; ++it0, ++it1, ++it_mask) {
    if (*it_mask) {
      *it1 = current_id++;
      *write_to++ = *it0;
    } else
      *it1 = none_tag;
  }
  return write_to;
}

/// @ingroup core_algorithms
/// @brief Keep elements by mask with default none tag (data.size()).
/// @overload
template <typename Range0, typename Range1, typename Range2>
auto keep_by_mask_and_make_map(Range0 &data, const Range1 &mask, Range2 &map) {
  using Index = std::decay_t<decltype(map[0])>;
  return keep_by_mask_and_make_map(data, mask, map, Index(data.size()));
}
} // namespace tf
