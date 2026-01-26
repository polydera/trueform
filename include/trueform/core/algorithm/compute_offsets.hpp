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

#include <functional>
namespace tf {

/// @ingroup core_algorithms
/// @brief Compute group offsets from sorted data.
///
/// Writes start offsets of groups where consecutive elements differ.
///
/// @tparam Range0 The input range type.
/// @tparam OutIter The output iterator type.
/// @tparam Val The offset value type.
/// @tparam Compare Binary predicate for equality comparison.
/// @param data Sorted input data.
/// @param out_iter Output iterator for offsets.
/// @param initial_offset Starting offset value.
/// @param compare Equality comparison predicate.
/// @return Iterator past the last written offset.
template <typename Range0, typename OutIter, typename Val, typename Compare>
auto compute_offsets(const Range0 &data, OutIter out_iter, Val initial_offset,
                     Compare &&compare) {
  if (!data.size())
    return out_iter;
  *out_iter++ = initial_offset++;
  auto iter = data.begin();
  auto end = data.end();
  auto current = iter;
  while (++iter != end) {
    if (!compare(*current, *iter)) {
      *out_iter++ = initial_offset;
      current = iter;
    }
    ++initial_offset;
  }
  *out_iter++ = initial_offset;
  return out_iter;
}

/// @ingroup core_algorithms
/// @brief Compute group offsets using equality comparison.
/// @overload
template <typename Range0, typename OutIter, typename Val>
auto compute_offsets(const Range0 &data, OutIter out_iter, Val initial_offset) {
  return compute_offsets(data, out_iter, initial_offset, std::equal_to<>{});
}
} // namespace tf
