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
#include "../views/zip.hpp"
#include "./parallel_for_each.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Parallel scatter copy using index map with none values.
///
/// Copies elements from src to dst at indices specified by map,
/// skipping elements where map equals none.
///
/// @tparam Range0 The source range type.
/// @tparam Range1 The destination range type.
/// @tparam Range2 The map range type.
/// @tparam Index The index type.
/// @param src Source elements.
/// @param dst Destination range.
/// @param map Index map (src[i] -> dst[map[i]]).
/// @param none Value indicating elements to skip.
template <typename Range0, typename Range1, typename Range2, typename Index>
auto parallel_copy_by_map_with_nones(const Range0 &src, Range1 &&dst,
                                     const Range2 &map, Index none) {
  tf::parallel_for_each(tf::zip(src, map), [&](auto pair) {
    auto &[_in, _id] = pair;
    if (_id != none)
      dst[_id] = _in;
  });
}

/// @ingroup core_algorithms
/// @brief Parallel scatter copy with default none value (map.size()).
/// @overload
template <typename Range0, typename Range1, typename Range2>
auto parallel_copy_by_map_with_nones(const Range0 &src, Range1 &&dst,
                                     const Range2 &map) {
  using Index = std::decay_t<decltype(map[0])>;
  parallel_copy_by_map_with_nones(src, dst, map, Index(map.size()));
}
} // namespace tf
