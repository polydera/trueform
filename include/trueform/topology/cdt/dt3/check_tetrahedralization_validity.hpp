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
#include "../../../core/buffer.hpp"
#include "../../../exact/orient3d.hpp"
#include "./tet_facets.hpp"
#include "./tet_state.hpp"
#include "./tetrahedralization_owner.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Check cell orientation, site coverage, reciprocal facet incidence and a
/// closed hull. This does not certify the Delaunay empty-sphere property.
template <typename Owner>
auto check_tetrahedralization_validity(const Owner &owner) -> bool {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;

  struct facet_record {
    Index a;
    Index b;
    Index c;
    Index tet;
    Index slot;
  };

  const std::size_t n = owner._corners.size();
  const std::size_t n_sites = owner._sites.size();
  if (n == 0 || n_sites == 0)
    return false;
  if (owner._hint < 0 || std::size_t(owner._hint) >= n)
    return false;

  tf::buffer<facet_record> facets;
  tf::buffer<std::array<Index, 4>> cells;
  tf::buffer<std::array<Index, 2>> rim;
  tf::buffer<char> covered;
  facets.reserve(n * 4);
  cells.reserve(n);
  covered.allocate_and_initialize(n_sites, char(0));

  std::size_t n_finite = 0;
  for (std::size_t tet = 0; tet < n; ++tet) {
    const auto state = owner._states[tet];
    if (state == tet_state::dead)
      return false;
    const auto corners = owner._corners[tet];
    const bool hull = state == tet_state::infinite;
    if (hull != (corners[3] == Owner::infinite))
      return false;
    if (!hull) {
      if (tet != n_finite)
        return false;
      ++n_finite;
    }

    for (std::size_t i = 0; i < 4; ++i) {
      if (corners[i] == Owner::infinite) {
        if (i != 3)
          return false;
        continue;
      }
      if (corners[i] < 0 || std::size_t(corners[i]) >= n_sites)
        return false;
      for (std::size_t j = i + 1; j < 4; ++j)
        if (corners[i] == corners[j])
          return false;
      if (!hull)
        covered[std::size_t(corners[i])] = 1;
    }

    if (!hull && !(tf::exact::orient3d_value_scaled<Int>(
                       owner._sites[std::size_t(corners[0])].pt,
                       owner._sites[std::size_t(corners[1])].pt,
                       owner._sites[std::size_t(corners[2])].pt,
                       owner._sites[std::size_t(corners[3])].pt) > 0))
      return false;
    if (hull) {
      rim.push_back({corners[0], corners[1]});
      rim.push_back({corners[1], corners[2]});
      rim.push_back({corners[2], corners[0]});
    }

    std::array<Index, 4> key{corners[0], corners[1], corners[2], corners[3]};
    std::sort(key.begin(), key.end());
    cells.push_back(key);
    for (std::size_t slot = 0; slot < 4; ++slot) {
      auto facet = tet_facet(corners, slot);
      std::sort(facet.begin(), facet.end());
      facets.push_back({facet[0], facet[1], facet[2], Index(tet), Index(slot)});
    }
  }
  if (n_finite != owner._n_finite)
    return false;
  for (std::size_t i = 0; i < n_sites; ++i)
    if (!covered[i])
      return false;

  std::sort(cells.begin(), cells.end());
  if (std::adjacent_find(cells.begin(), cells.end()) != cells.end())
    return false;

  std::sort(facets.begin(), facets.end(),
            [](const facet_record &x, const facet_record &y) {
              return x.a < y.a ||
                     (x.a == y.a && (x.b < y.b || (x.b == y.b && x.c < y.c)));
            });
  for (std::size_t i = 0; i < facets.size(); i += 2) {
    const auto &x = facets[i];
    const auto &y = facets[i + 1];
    if (x.a != y.a || x.b != y.b || x.c != y.c)
      return false;
    if (i + 2 < facets.size() && facets[i + 2].a == x.a &&
        facets[i + 2].b == x.b && facets[i + 2].c == x.c)
      return false;
    if (owner._neighbors[std::size_t(x.tet)][std::size_t(x.slot)] != y.tet ||
        owner._neighbors[std::size_t(y.tet)][std::size_t(y.slot)] != x.tet)
      return false;
    const bool x_hull =
        owner._states[std::size_t(x.tet)] == tet_state::infinite;
    const bool y_hull =
        owner._states[std::size_t(y.tet)] == tet_state::infinite;
    if ((x.a == Owner::infinite) != (x_hull && y_hull))
      return false;
  }

  std::sort(rim.begin(), rim.end());
  if (std::adjacent_find(rim.begin(), rim.end()) != rim.end())
    return false;
  for (auto &edge : rim)
    if (edge[1] < edge[0])
      std::swap(edge[0], edge[1]);
  std::sort(rim.begin(), rim.end());
  if (rim.size() % 2 != 0)
    return false;
  for (std::size_t i = 0; i < rim.size(); i += 2)
    if (rim[i] != rim[i + 1] || (i + 2 < rim.size() && rim[i + 2] == rim[i]))
      return false;
  return true;
}

} // namespace tf::topology::cdt::dt3
