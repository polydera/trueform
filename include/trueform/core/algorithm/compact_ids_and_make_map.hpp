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

#include "../buffer.hpp"
#include "../range.hpp"
#include "./parallel_fill.hpp"

#include <cstddef>
#include <utility>

namespace tf {

/// @ingroup core_algorithms
/// @brief Compact a bounded mutable ID stream and build its old-to-new map.
///
/// Active IDs receive dense IDs in first-occurrence order and are rewritten in
/// place. Inactive map entries retain the sentinel `old_to_new.size()`.
///
/// @param ids Mutable IDs in `[0, old_to_new.size())`.
/// @param old_to_new Pre-sized scratch for the bounded source ID universe.
/// @return Number of active compact IDs.
template <typename IdIterator, std::size_t IdSize, typename MapIterator,
          std::size_t MapSize>
auto compact_ids_and_make_map(tf::range<IdIterator, IdSize> ids,
                              tf::range<MapIterator, MapSize> old_to_new) {
  using Index = typename tf::range<MapIterator, MapSize>::value_type;
  const Index sentinel = static_cast<Index>(old_to_new.size());
  tf::parallel_fill(old_to_new, sentinel);

  Index next_id = 0;
  for (auto &&id : ids) {
    Index &mapped = old_to_new[static_cast<std::size_t>(id)];
    if (mapped == sentinel)
      mapped = next_id++;
    id = mapped;
  }
  return next_id;
}

/// @ingroup core_algorithms
/// @brief Compact IDs using an owning old-to-new map.
/// @overload
template <typename IdIterator, std::size_t IdSize, typename Index>
auto compact_ids_and_make_map(tf::range<IdIterator, IdSize> ids,
                              Index bounded_id_count)
    -> std::pair<Index, tf::buffer<Index>> {
  tf::buffer<Index> old_to_new;
  old_to_new.allocate(static_cast<std::size_t>(bounded_id_count));
  auto compact_id_count =
      tf::compact_ids_and_make_map(ids, tf::make_range(old_to_new));
  return {compact_id_count, std::move(old_to_new)};
}

} // namespace tf
