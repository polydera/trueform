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
/// @brief Remove elements and create mapping to new indices.
///
/// Removes elements matching predicate while building a map from
/// old indices to new compact indices.
///
/// @tparam Range0 The data range type.
/// @tparam F Predicate type (returns true for elements to remove).
/// @tparam Range1 The map range type.
/// @tparam Index The index type.
/// @param data Data to filter (modified in-place).
/// @param predicate Returns true for elements to remove.
/// @param map Output map from old to new indices.
/// @param none_tag Value for removed element mappings.
/// @return Iterator to new end of data.
template <typename Range0, typename F, typename Range1, typename Index>
auto remove_if_and_make_map(Range0 &data, const F &predicate, Range1 &map,
                            Index none_tag) {
  Index current_id = 0;
  auto it0 = data.begin();
  auto write_to = it0;
  auto end0 = data.end();
  auto it1 = map.begin();
  for (; it0 != end0; ++it0, ++it1) {
    if (!predicate(*it0)) {
      *it1 = current_id++;
      *write_to++ = *it0;
    } else
      *it1 = none_tag;
  }
  return write_to;
}

/// @ingroup core_algorithms
/// @brief Remove elements with default none tag (data.size()).
/// @overload
template <typename Range0, typename F, typename Range1>
auto remove_if_and_make_map(Range0 &data, const F &predicate, Range1 &map) {
  using Index = std::decay_t<decltype(map[0])>;
  return remove_if_and_make_map(data, predicate, map, Index(data.size()));
}
} // namespace tf
