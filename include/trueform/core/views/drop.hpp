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
#include "../range.hpp"

namespace tf {
/// @ingroup core_ranges
/// @brief Creates a view that skips the first `n` elements of a range.
///
/// Returns a subrange starting after the first `n` elements. If the range
/// has fewer than `n` elements, behavior is undefined.
///
/// @tparam Range The input range type.
/// @param range The input range.
/// @param n The number of elements to skip.
/// @return A view over the remaining elements after skipping `n`.
///
/// @see @ref tf::take
/// @see @ref tf::slice
template <typename Range> auto drop(Range &&range, std::size_t n) {
  return tf::make_range(range.begin() + n, range.end());
}
} // namespace tf
