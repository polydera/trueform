/*
 * Copyright (c) 2026 XLAB
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

#include "../../core/buffer.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

/// @ingroup cut
/// @brief Whether `key` already carries a split at `parameter`.
///
/// Two splits are the same split when they land on the same point of the
/// grid the transported parameter defines, or on neighbouring ones. The
/// tolerance is one quantum of that grid, so it follows the width by
/// construction rather than through a second constant of its own.
///
/// The question is asked while splits are being produced, so it spans two
/// tables: `sorted`, which answers by binary search, and `fresh`, the ones
/// this wave has appended and nobody has ordered yet. Searching an appended
/// table as if it were ordered is unsound in both directions — it can return
/// a span of unrelated edges, admitting a duplicate onto one edge, or
/// suppress a real split against one that is not on this edge at all.
template <typename Index, typename Split>
auto has_adjacent_region_split(
    const std::array<std::array<Index, 2>, 2> &key,
    decltype(Split{}.param) parameter, const tf::buffer<Split> &sorted,
    const tf::buffer<Split> &fresh) -> bool {
  using param_t = decltype(Split{}.param);
  auto adjacent = [parameter](const Split &split) {
    const param_t distance = split.param - parameter;
    return distance <= param_t(1) && distance >= param_t(-1);
  };
  const auto range = tf::cut::detail::find_region_edge_splits(key, sorted);
  for (Index split = range.first; split < range.second; ++split)
    if (adjacent(sorted[std::size_t(split)]))
      return true;
  for (const auto &split : fresh)
    if (split.edge == key && adjacent(split))
      return true;
  return false;
}

/// @ingroup cut
/// @brief Whether `key` already carries a split at `parameter`, over one
///        ordered table.
template <typename Index, typename Split>
auto has_adjacent_region_split(
    const std::array<std::array<Index, 2>, 2> &key,
    decltype(Split{}.param) parameter, const tf::buffer<Split> &sorted)
    -> bool {
  return has_adjacent_region_split<Index>(key, parameter, sorted,
                                          tf::buffer<Split>{});
}

} // namespace tf::cut
