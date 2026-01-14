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

#include "../iter/zip_iterator.hpp"
#include "../range.hpp"

namespace tf {
/// @ingroup core_ranges
/// @brief Creates a view that iterates over multiple ranges in parallel.
///
/// Returns a range where each element is a tuple of corresponding elements
/// from all input ranges. The resulting range has the length of the shortest
/// input range.
///
/// @tparam Range0 The first range type.
/// @tparam Range1 The second range type.
/// @tparam Ranges Additional range types.
/// @param r0 The first range.
/// @param r1 The second range.
/// @param r Additional ranges.
/// @return A view of tuples containing corresponding elements from all ranges.
template <typename Range0, typename Range1, typename... Ranges>
auto zip(Range0 &&r0, Range1 &&r1, Ranges &&...r) {
  return tf::make_range(
      iter::make_zip_iterator(r0.begin(), r1.begin(), r.begin()...),
      iter::make_zip_iterator(r0.end(), r1.end(), r.end()...));
}
} // namespace tf
