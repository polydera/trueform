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
#include "./delaunay_frontier.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>

namespace tf::topology::cdt {

struct morton_partition {
  std::size_t cut;
  delaunay_axis axis;
};

inline auto highest_morton_plane(std::uint32_t changed) -> unsigned {
  unsigned plane = 0;
  while (changed > 1) {
    changed >>= 1U;
    ++plane;
  }
  return plane;
}

/// Split one sorted Morton interval at its highest changing bit. A terminal
/// interval is represented by an empty optional, so callers cannot
/// accidentally consume a zero cut or a stale axis.
template <typename KeyBuffer>
auto partition_morton_sites(const KeyBuffer &keys, std::size_t first,
                            std::size_t last)
    -> std::optional<morton_partition> {
  const std::uint32_t changed = keys[first] ^ keys[last - 1];
  if (changed == 0)
    return std::nullopt;

  const unsigned plane = highest_morton_plane(changed);
  const std::uint32_t plane_mask = std::uint32_t(1) << plane;
  const std::uint32_t first_upper_key = keys[last - 1] & ~(plane_mask - 1U);
  const auto begin = keys.begin() + std::ptrdiff_t(first);
  const auto end = keys.begin() + std::ptrdiff_t(last);
  const auto boundary = std::lower_bound(begin, end, first_upper_key);
  const std::size_t cut =
      first + static_cast<std::size_t>(std::distance(begin, boundary));
  if (cut - first < 2 || last - cut < 2)
    return std::nullopt;

  return morton_partition{cut, (plane & 1U) == 0 ? delaunay_axis::x
                                                 : delaunay_axis::y};
}

} // namespace tf::topology::cdt
