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

#include "../iter/blocked_iterator.hpp"
#include "../range.hpp"

namespace tf {

/// @ingroup core_ranges
/// @brief Create a view grouping elements into consecutive blocks.
///
/// Interprets flat buffers as structured collections (e.g., every 3 elements
/// form a triangle). The resulting range yields blocks as subranges.
///
/// @tparam Range The input range type (must support random access).
/// @param r The input range to be grouped.
/// @param block_size The number of elements per block.
/// @return A view with elements grouped into fixed-size blocks.
///
/// @note This overload uses a runtime-specified block size.
template <typename Range>
auto make_blocked_range(Range &&r, std::size_t block_size) {
  auto begin = tf::iter::make_blocked_iterator(r.begin(), block_size);
  auto end = tf::iter::make_blocked_iterator(r.end(), block_size);
  return tf::make_range(std::move(begin), std::move(end));
}

/// @ingroup core_ranges
/// @brief Create a view grouping elements with compile-time block size.
///
/// Uses a static block size enabling optimizations and @ref tf::static_size
/// propagation. Each block has `BlockSize` elements accessible via
/// structured bindings.
///
/// @tparam BlockSize The elements per block (compile-time constant).
/// @tparam Range The input range type.
/// @param r The input range to be grouped.
/// @return A view with @ref tf::static_size set to `BlockSize`.
template <std::size_t BlockSize, typename Range>
auto make_blocked_range(Range &&r) {
  auto begin = tf::iter::make_blocked_iterator<BlockSize>(r.begin());
  auto end = tf::iter::make_blocked_iterator<BlockSize>(r.end());
  return tf::make_range(std::move(begin), std::move(end));
}

} // namespace tf
