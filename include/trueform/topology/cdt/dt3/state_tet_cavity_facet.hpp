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
#include "../../../core/config/inline.hpp"
#include "./overflow_tet_cavity_links.hpp"
#include "./tet_cavity_links.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Index>
TF_FORCE_INLINE auto state_tet_cavity_facet(tet_cavity_links<Index> &links,
                                            const std::array<Index, 3> &facet)
    -> void {
  constexpr std::size_t max_local = tet_cavity_links<Index>::max_local;
  const std::size_t i = links.n_facets;
  if (i == links.facets.size()) {
    const std::size_t grown = i + i / 2 + 64;
    links.facets.reallocate(grown);
    links.across.reallocate(3 * grown);
  }
  links.facets[i] = facet;
  links.n_facets = i + 1;
  if (links.overflowed)
    return;
  std::uint8_t *stamps[3];
  std::size_t local[3];
  bool fresh[3];
  // The count of locals is carried in a register: a stamp is a byte, and a
  // byte store sends every count read through memory back to it.
  std::size_t n_local = links.n_local;
  for (std::size_t q = 0; q < 3; ++q) {
    stamps[q] = &links.local_of_site[tet_cavity_stamp_slot(facet[q])];
    local[q] = *stamps[q];
    fresh[q] =
        (local[q] >= n_local) | (links.site_of_local[local[q]] != facet[q]);
  }
  // An id past the last local is the overflow: its byte and its name are
  // read by no one before the next open.
  for (std::size_t q = 0; q < 3; ++q) {
    local[q] = fresh[q] ? n_local : local[q];
    n_local += fresh[q];
    *stamps[q] = std::uint8_t(local[q]);
    links.site_of_local[local[q]] = facet[q];
  }
  links.n_local = n_local;
  if (n_local > max_local) {
    overflow_tet_cavity_links(links, i);
    return;
  }
  for (std::size_t e = 0; e < 3; ++e) {
    const std::size_t a = local[(e + 1) % 3], b = local[(e + 2) % 3];
    const std::size_t lo = a < b ? a : b, hi = a < b ? b : a;
    auto &slot = links.table[lo * max_local + hi];
    const Index held = slot, mine = Index(3 * i + e);
    links.across[std::size_t(mine)] = held;
    *(held < 0 ? &links.sink : &links.across[std::size_t(held)]) = mine;
    slot = held < 0 ? mine : Index(-1);
  }
}

} // namespace tf::topology::cdt::dt3
