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
#include "../iter/mapped_iterator.hpp"
#include "../range.hpp"
namespace tf {
namespace views {
template <typename Range> struct offset_block_policy {
  Range range;
  template <typename Iterator> auto operator()(Iterator start) const {
    return tf::make_range(range.begin() + *start,
                          range.begin() + *std::next(start));
  }
};

template <typename Range0, typename Range1>
auto make_offset_block_begin(Range0 &&offsets, Range1 &&data) {
  auto data_view = tf::make_range(data);
  return tf::iter::make_iter_mapped(
      offsets.begin(),
      views::offset_block_policy<decltype(data_view)>{data_view});
}

template <typename Range0, typename Range1>
auto make_offset_block_end(Range0 &&offsets, Range1 &&data) {
  auto data_view = tf::make_range(data);
  return tf::iter::make_iter_mapped(
      offsets.size() ? std::prev(offsets.end()) : offsets.end(),
      views::offset_block_policy<decltype(data_view)>{data_view});
}
} // namespace views

/// @ingroup core_ranges
/// @brief Creates a range view over a sequence of blocks defined by offsets.
///
/// This utility allows you to iterate over subranges ("blocks") of a data
/// range. The start and end of each block is defined by two consecutive
/// elements in the `offsets` range. The number of blocks is thus
/// `offsets.size() - 1`.
///
/// This is especially useful for grouping elements like polygons, faces, or
/// variable-sized clusters, where each block may contain a different number of
/// elements.
///
/// @tparam Range0 A range of integral offsets (e.g. `std::vector<size_t>`).
/// @tparam Range1 A range of underlying data values.
/// @param offsets A range of offsets. Must contain `n + 1` elements to define
/// `n` blocks.
/// @param data A range of elements from which the blocks are constructed.
/// @return A range view over blocks of data, where each block is itself a
/// `tf::range`.
///
/// @code
/// tf::buffer<std::size_t> offsets{0, 3, 6};
/// tf::buffer<int> values{1, 2, 3, 4, 5, 6};
/// for (auto block : make_offset_block_range(offsets, values)) {
///   for (int v : block) std::cout << v << ' ';
///   std::cout << '\n';
/// }
/// // Output:
/// // 1 2 3
/// // 4 5 6
/// @endcode
///
/// @note This function returns a view — it does not copy the underlying data.
template <typename Range0, typename Range1>
auto make_offset_block_range(Range0 &&offsets, Range1 &&data) {
  return tf::make_range(views::make_offset_block_begin(offsets, data),
                        views::make_offset_block_end(offsets, data));
}
} // namespace tf
