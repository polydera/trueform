/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once

namespace tf {
/// @ingroup algorithms
/// @brief Circularly decrements an index within a fixed range.
///
/// Given an index `val` in the range `[0, end)`, returns `val - 1` if `val >
/// 0`, and `end - 1` if `val == 0` (wrap-around).
///
///
/// @tparam Index An unsigned integer type.
/// @param val The current index. Must satisfy `0 <= val < end`.
/// @param end The exclusive upper bound of the range (must be > 0).
/// @return `(val - 1 + end) % end`, computed without modulo.
template <typename Index> auto circular_decrement(Index val, Index end) {
  return (end - 1) * (val == 0) + (val != 0) * (val - 1);
}

} // namespace tf
