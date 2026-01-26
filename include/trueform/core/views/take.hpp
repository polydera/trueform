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
/// @brief Creates a view over the first `n` elements of a range.
///
/// Returns a subrange containing only the first `n` elements. If the range
/// has fewer than `n` elements, behavior is undefined.
///
/// @tparam Range The input range type.
/// @param range The input range.
/// @param n The number of elements to take.
/// @return A view over the first `n` elements.
///
/// @see @ref tf::drop
/// @see @ref tf::slice
template <typename Range> auto take(Range &&range, std::size_t n) {
  return tf::make_range(range.begin(), range.begin() + n);
}
} // namespace tf
