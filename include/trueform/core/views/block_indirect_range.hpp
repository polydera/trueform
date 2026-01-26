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
#include "./indirect_range.hpp"
#include "./mapped_range.hpp"
#include "../range.hpp"
namespace tf {
namespace views {
template <typename Range> struct block_indirect_policy {
  Range range;
  template <typename Range1> auto operator()(Range1 &&ids) const {
    return tf::make_indirect_range(ids, range);
  }
};
} // namespace implementation

/// @ingroup core_ranges
/// @brief Creates a range of indirect views into a data range based on block-wise indexing.
///
/// This utility takes a range of blocks, where each block is itself a range of indices into
/// a separate data range. It returns a range of `tf::indirect_range` objects that allow
/// accessing the actual values in the data range based on those indices.
///
/// This is commonly used for cases like triangle meshes where you store a list of triangle vertex indices,
/// and wish to access the vertex positions directly.
///
/// The static size of blocks (e.g., via @ref tf::make_blocked_range<3>) is preserved and propagated,
/// enabling structured bindings and compile-time dimension checks.
///
/// @tparam Range0 A range of index blocks 
/// @tparam Range1 A range of underlying data values.
/// @param blocks A range where each element is itself a small range of indices into the data.
/// @param data A range of actual data values (e.g., vertices).
/// @return A view over indirect subranges into `data`, one per block in `blocks`.
///
/// @code
/// tf::buffer<std::size_t> indices{0, 1, 2, 2, 3, 0};
/// auto triangles = tf::make_blocked_range<3>(indices);
/// tf::buffer<tf::vector<float, 3>> vertices = {
///     {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}};
///
/// for (auto triangle : tf::make_block_indirect_range(triangles, vertices)) {
///   auto [i0, i1, i2] = triangle.ids();  // Structured binding over block indices
///   auto [v0, v1, v2] = triangle;        // Structured binding over the vertex positions
///   std::cout << "Triangle (ids: " << i0 << ", " << i1 << ", " << i2 << "): ";
/// }
/// @endcode
///
/// @note This function returns a view — it does not copy the underlying data.
template <typename Range0, typename Range1>
auto make_block_indirect_range(Range0 &&blocks, Range1 &&data) {
  auto data_view = tf::make_range(data);
  return tf::make_mapped_range(
      static_cast<Range0 &&>(blocks),
      views::block_indirect_policy<decltype(data_view)>{
          std::move(data_view)});
}
} // namespace tf
