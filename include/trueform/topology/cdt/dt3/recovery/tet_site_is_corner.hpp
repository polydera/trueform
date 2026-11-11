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
#include "./tet_segment_recovery.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether the site `s` is a corner of the input: a site no tier of the
/// recovery minted, other than an end of the segment `except`. O(1).
template <typename Index, typename Coord>
auto tet_site_is_corner(const tet_segment_recovery<Index, Coord> &recovery,
                        Index s, Index except = Index(-1)) -> bool {
  if (std::size_t(s) < recovery.steiner.size() &&
      recovery.steiner[std::size_t(s)])
    return false;
  if (except == Index(-1))
    return true;
  const auto &rows = recovery.subsegments;
  return rows[std::size_t(recovery.first[std::size_t(except)])].a != s &&
         rows[std::size_t(recovery.last[std::size_t(except)])].b != s;
}

} // namespace tf::topology::cdt::dt3
