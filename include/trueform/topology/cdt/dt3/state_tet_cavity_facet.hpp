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
#include "./overflow_tet_cavity_links.hpp"
#include "./tet_cavity_links.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Index>
auto state_tet_cavity_facet(tet_cavity_links<Index> &links,
                            const std::array<Index, 3> &facet) -> void {
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
  const std::uint32_t generation = links.generation << 8;
  std::uint32_t local[3];
  for (std::size_t q = 0; q < 3; ++q) {
    auto &stamp = links.local_of_site[tet_cavity_stamp_slot(facet[q])];
    const bool fresh = (stamp & ~std::uint32_t(255)) != generation;
    const std::uint32_t l =
        fresh ? std::uint32_t(links.n_local) : (stamp & 255u);
    links.n_local += fresh;
    // The local field is eight bits: the id past it is the overflow and must
    // not carry into the generation, or the next flood reads it as its own.
    stamp = generation | (l & 255u);
    local[q] = l;
  }
  if (links.n_local > max_local) {
    overflow_tet_cavity_links(links, i);
    return;
  }
  for (std::size_t e = 0; e < 3; ++e) {
    const std::uint32_t a = local[(e + 1) % 3], b = local[(e + 2) % 3];
    const std::uint32_t lo = a < b ? a : b, hi = a < b ? b : a;
    auto &slot = links.table[std::size_t(lo) * max_local + std::size_t(hi)];
    const Index held = slot, mine = Index(3 * i + e);
    links.across[std::size_t(mine)] = held;
    *(held < 0 ? &links.sink : &links.across[std::size_t(held)]) = mine;
    slot = held < 0 ? mine : Index(-1);
  }
}

} // namespace tf::topology::cdt::dt3
