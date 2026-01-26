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
/// @brief Creates a view over a subrange from index `_from` to `_to`.
///
/// Returns a subrange containing elements at indices `[_from, _to)`.
/// Equivalent to combining @ref tf::drop and @ref tf::take.
///
/// @tparam Range The input range type.
/// @param range The input range.
/// @param _from The starting index (inclusive).
/// @param _to The ending index (exclusive).
/// @return A view over elements `[_from, _to)`.
///
/// @see @ref tf::take
/// @see @ref tf::drop
template <typename Range>
auto slice(Range &&range, std::size_t _from, std::size_t _to) {
  return tf::make_range(range.begin() + _from, range.begin() + _to);
}
} // namespace tf
