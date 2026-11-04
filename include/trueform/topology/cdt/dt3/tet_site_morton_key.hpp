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
#include "./tet_morton_code.hpp"
#include "./tet_site_domain.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The shift every axis of `domain` takes alike so its widest span fits a
/// Morton code.
template <typename Coord>
auto tet_site_morton_shift(const tet_site_domain<Coord> &domain) -> unsigned {
  auto remaining =
      std::max(domain.span[0], std::max(domain.span[1], domain.span[2])) >>
      tet_morton_bits;
  unsigned shift = 0;
  while (remaining != Coord(0)) {
    ++shift;
    remaining = remaining >> 1U;
  }
  return shift;
}

/// The Morton key of `pt` within `domain` under its `shift`: the spatial
/// order the weld stores sites in and an append inserts them in.
template <typename Coord, typename Point>
auto tet_site_morton_key(const Point &pt, const tet_site_domain<Coord> &domain,
                         unsigned shift) -> std::uint64_t {
  return tet_morton_code(std::uint32_t(domain.offset(0, pt[0]) >> shift),
                         std::uint32_t(domain.offset(1, pt[1]) >> shift),
                         std::uint32_t(domain.offset(2, pt[2]) >> shift));
}

} // namespace tf::topology::cdt::dt3
