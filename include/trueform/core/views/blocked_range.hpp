/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../iter/blocked_iterator.hpp"
#include "../range.hpp"

namespace tf {

/// @ingroup ranges
/// @brief Creates a view over a range where elements are grouped into
/// consecutive blocks.
///
/// This is useful when interpreting flat buffers as structured collections,
/// such as interpreting a buffer of indices where every 3 consecutive elements
/// form a triangle.
///
/// The resulting range is a non-owning view that yields blocks of size
/// `block_size`, each presented as a subrange. For example, a buffer `[0, 1, 2,
/// 3, 4, 5]` with `block_size = 3` yields two blocks: `[0, 1, 2]` and `[3, 4,
/// 5]`.
///
/// @tparam Range The type of the input range (must support random access).
/// @param r The input range to be grouped.
/// @param block_size The number of elements in each group.
/// @return A view over the range with elements grouped into fixed-size blocks.
///
/// @note This overload accepts a dynamic block size specified at runtime.
template <typename Range>
auto make_blocked_range(Range &&r, std::size_t block_size) {
  auto begin = tf::iter::make_blocked_iterator(r.begin(), block_size);
  auto end = tf::iter::make_blocked_iterator(r.end(), block_size);
  return tf::make_range(std::move(begin), std::move(end));
}

/// @ingroup ranges
/// @brief Creates a view over a range where elements are grouped into
/// consecutive blocks, with compile-time block size.
///
/// This overload provides a static block size via the template parameter
/// `BlockSize`, which can enable additional optimizations and compile-time
/// checks.
///
/// The resulting range is a non-owning view that yields blocks of size
/// `BlockSize`, each presented as a subrange. For example, a buffer `[0, 1, 2,
/// 3, 4, 5]` with `BlockSize = 3` yields two blocks: `[0, 1, 2]` and `[3, 4,
/// 5]`.
///
/// @tparam BlockSize The number of elements per block (known at compile time).
/// @tparam Range The type of the input range.
/// @param r The input range to be grouped.
/// @return A view over the input range with elements grouped into blocks of
/// size `BlockSize`.
///
/// @note This version propagates static size metadata where possible.
template <std::size_t BlockSize, typename Range>
auto make_blocked_range(Range &&r) {
  auto begin = tf::iter::make_blocked_iterator<BlockSize>(r.begin());
  auto end = tf::iter::make_blocked_iterator<BlockSize>(r.end());
  return tf::make_range(std::move(begin), std::move(end));
}

} // namespace tf
