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
#include "../sort_tet_corner_key.hpp"
#include "../tet_deepest_of.hpp"
#include "../tet_facets.hpp"
#include "../tet_finite_cell.hpp"
#include "../tet_orient3d_sites.hpp"
#include "../tet_with_sites.hpp"
#include "./tet_segment_encroacher.hpp"
#include "./tet_site_on_segment.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Read what the missing `row` meets along the cells its segment pierces,
/// `star` the star of its first end: the first site strictly on it into
/// `query.on`, and short of one the encroacher of the pierced cells' sites
/// (@ref offer_tet_segment_encroacher), each site read with the row's ends at
/// the deepest depth of the three. The walk starts in the star cell
/// whose facet opposite the first end holds the second in its cone, the
/// least sorted-corner key among ties, and leaves each cell through the
/// facet its line pierces, an edge or corner passed through counting; it
/// ends at the second end, at the hull, or where no facet is pierced.
template <typename DT, typename Star, typename Row, typename Query>
auto meet_tet_subsegment(const DT &dt, const Star &star, const Row &row,
                         Query &query) -> void {
  using Index = typename DT::index_type;
  const auto &owner = dt.owner();
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  const Index a = row.a, b = row.b;
  const auto orient = [&owner](Index p, Index q, Index r, Index s) {
    return tet_orient3d_sites(owner, p, q, r, s);
  };
  const auto meets = [&](Index corner) {
    if (corner == a || corner == b)
      return false;
    return tet_with_sites(
        owner, tet_deepest_of(owner, a, b, corner), [&](const auto &sites) {
          using lattice = typename std::decay_t<decltype(sites)>::int_type;
          const auto pa = sites[std::size_t(a)].pt;
          const auto pb = sites[std::size_t(b)].pt;
          const auto pt = sites[std::size_t(corner)].pt;
          if (tet_site_on_segment<lattice>(pa, pb, pt)) {
            query.on = corner;
            return true;
          }
          offer_tet_segment_encroacher(pa, pb, pt, corner, query.encroacher,
                                       query.cosine);
          return false;
        });
  };
  Index cell = DT::k_none;
  std::size_t slot = 0;
  std::array<Index, 4> least{};
  for (auto candidate : star) {
    if (!tet_finite_cell(dt, Index(candidate)))
      continue;
    const auto corners = cells[std::size_t(candidate)];
    std::size_t at = 0;
    while (corners[at] != a)
      ++at;
    const auto facet = tet_facet(corners, at);
    bool held = true;
    for (std::size_t i = 0; i < 3 && held; ++i)
      held = orient(a, facet[i], facet[(i + 1) % 3], b) >= 0;
    if (!held)
      continue;
    std::array<Index, 4> key{Index(corners[0]), Index(corners[1]),
                             Index(corners[2]), Index(corners[3])};
    sort_tet_corner_key(key);
    if (cell == DT::k_none || key < least) {
      least = key;
      cell = Index(candidate);
      slot = at;
    }
  }
  if (cell == DT::k_none) {
    for (auto c : star)
      for (auto corner : cells[std::size_t(c)])
        if (meets(Index(corner)))
          return;
    return;
  }
  for (auto corner : cells[std::size_t(cell)])
    if (meets(Index(corner)))
      return;
  for (std::size_t step = 0; step < dt.n_tets(); ++step) {
    const Index other = neighbors[std::size_t(cell)][slot];
    if (!tet_finite_cell(dt, other))
      return;
    const auto facet = tet_facet(cells[std::size_t(cell)], slot);
    const auto across = cells[std::size_t(other)];
    std::size_t at = 0;
    while (across[at] == facet[0] || across[at] == facet[1] ||
           across[at] == facet[2])
      ++at;
    const Index w = Index(across[at]);
    if (w == b || meets(w))
      return;
    std::size_t exit = 4, loose = 4;
    for (std::size_t t = 0; t < 4 && exit == 4; ++t) {
      if (t == at)
        continue;
      const auto g = tet_facet(across, t);
      int positive = 0, negative = 0;
      for (std::size_t k = 0; k < 3; ++k) {
        const int s = orient(a, b, Index(g[k]), Index(g[(k + 1) % 3]));
        positive += s > 0;
        negative += s < 0;
      }
      if (positive && negative)
        continue;
      if (positive == 3 || negative == 3)
        exit = t;
      else if (positive + negative && loose == 4)
        loose = t;
    }
    if (exit == 4)
      exit = loose;
    if (exit == 4)
      return;
    cell = other;
    slot = exit;
  }
}

} // namespace tf::topology::cdt::dt3
