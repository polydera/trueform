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
#include "./tet_site_priority.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

inline constexpr std::size_t tet_site_levels = 32;

/// The insertion level of a site from its priority: the leading zero bit
/// pairs, so each level holds a quarter of the sites the one below it does
/// and the levels are the biased random insertion order's rounds, read from
/// the last to the first.
inline auto tet_site_level(std::uint64_t priority) -> std::size_t {
  std::size_t level = 0;
  while (level + 1 < tet_site_levels &&
         (priority >> (62U - 2U * unsigned(level))) == 0)
    ++level;
  return level;
}

} // namespace tf::topology::cdt::dt3
