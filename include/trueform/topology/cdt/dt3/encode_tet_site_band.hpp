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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_morton_code.hpp"
#include "./tet_site_domain.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Encode the band's Morton keys, every axis shifted alike so the domain's
/// widest span fits a code; no key outside the band is consumed by it.
template <bool Parallel = false, typename Owner>
auto encode_tet_site_band(
    Owner &owner, std::size_t begin, std::size_t end,
    const tet_site_domain<typename Owner::coord_type> &domain) -> void {
  using Coord = typename Owner::coord_type;
  auto remaining =
      std::max(domain.span[0], std::max(domain.span[1], domain.span[2])) >>
      tet_morton_bits;
  unsigned coordinate_shift = 0;
  while (remaining != Coord(0)) {
    ++coordinate_shift;
    remaining = remaining >> 1U;
  }

  owner._keys.allocate(owner._sites.size());
  const auto encode_site = [&owner, &domain, coordinate_shift](std::size_t i) {
    const auto &p = owner._sites[i].pt;
    owner._keys[i] = tet_morton_code(
        std::uint32_t(domain.offset(0, p[0]) >> coordinate_shift),
        std::uint32_t(domain.offset(1, p[1]) >> coordinate_shift),
        std::uint32_t(domain.offset(2, p[2]) >> coordinate_shift));
  };
  if constexpr (Parallel)
    tf::parallel_for_each(tf::make_sequence_range(begin, end), encode_site,
                          tf::checked);
  else
    for (std::size_t i = begin; i < end; ++i)
      encode_site(i);
}

} // namespace tf::topology::cdt::dt3
