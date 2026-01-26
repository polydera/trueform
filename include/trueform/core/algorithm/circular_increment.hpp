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

namespace tf {
/// @ingroup core_algorithms
/// @brief Circularly increments an index within a fixed range.
///
/// Given an index `val` in the range `[0, end)`, returns `val + 1` if it is less than `end - 1`,
/// and `0` if it would wrap around.
///
///
/// @tparam Index An unsigned integer type.
/// @param val The current index. Must satisfy `0 <= val < end`.
/// @param end The exclusive upper bound of the range (must be > 0).
/// @return `(val + 1) % end`, computed without modulo.
template <typename Index> auto circular_increment(Index val, Index end) {
  val += 1;
  return val * (val < end);
}
} // namespace tf
