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

#include "./sequence_range.hpp"
#include "./zip.hpp"

namespace tf {
/// @ingroup core_ranges
/// @brief Creates an enumerated view over a range, pairing each element with
/// its index.
///
/// Returns a zipped range of indices and elements, similar to Python's
/// `enumerate()`. Each element yields a tuple of (index, element).
///
/// @tparam Range The input range type.
/// @param r The input range to enumerate.
/// @return A view of (index, element) pairs.
///
/// @see @ref tf::zip
/// @see @ref tf::make_sequence_range
template <typename Range> auto enumerate(Range &&r) {
  return tf::zip(tf::make_sequence_range(r.size()), r);
}
} // namespace tf
