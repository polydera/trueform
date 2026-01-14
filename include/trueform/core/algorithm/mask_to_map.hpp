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
/// @brief Convert a boolean mask into an index mapping with explicit none-tag.
///
/// Given a boolean `mask` and a pre-allocated `map` of the same size,
/// this function assigns consecutive indices to `true` entries in the mask,
/// and assigns `none_tag` to entries that are `false`.
///
/// This is typically used to remap elements while preserving the mapping of
/// excluded elements to a sentinel value (e.g., for compacted arrays).
///
/// @tparam Range0 Type of the boolean mask (must support `.begin()`, `.end()`).
/// @tparam Range1 Type of the output mapping range (must support `.begin()`,
/// `.end()`).
/// @tparam Index Type of the index values.
/// @param mask A range of booleans indicating which elements to include.
/// @param map A range to be filled with the computed mapping (must be same size
/// as `mask`).
/// @param none_tag The value to assign to entries where the mask is `false`.
/// @return The number of `true` elements in the mask (i.e., the number of valid
/// mappings).
template <typename Range0, typename Range1, typename Index>
auto mask_to_map(const Range0 &mask, Range1 &&map, const Index &none_tag) {
  Index current_id = 0;
  auto it0 = mask.begin();
  auto end0 = mask.end();
  auto it1 = map.begin();
  for (; it0 != end0; ++it0, ++it1) {
    if (*it0)
      *it1 = current_id++;
    else
      *it1 = none_tag;
  }
  return current_id;
}

/// @ingroup core_algorithms
/// @brief Convert a boolean mask into an index mapping using default none-tag.
///
/// This overload infers the `none_tag` value as the size of the `map`,
/// and then delegates to the explicit overload of @ref mask_to_map.
///
/// @tparam Range0 Type of the boolean mask.
/// @tparam Range1 Type of the output mapping.
/// @param mask A range of booleans indicating which elements to include.
/// @param map A range to be filled with the computed mapping.
/// @return The number of `true` elements in the mask.
template <typename Range0, typename Range1>
auto mask_to_map(const Range0 &mask, Range1 &&map) {
  using index_t = std::decay_t<decltype(map[0])>;
  index_t none_tag = map.size();
  return mask_to_map(mask, map, none_tag);
}

} // namespace tf
