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
#include "../../../../exact/meta.hpp"
#include "../../../../exact/projection_axes.hpp"
#include "../state_tet_protected_facets.hpp"
#include "../tet_facet_is_protected.hpp"
#include "../tet_facet_key.hpp"
#include "../tet_orient3d_sign.hpp"
#include "../tet_shared_owners.hpp"
#include "../tet_site_position.hpp"
#include "../tet_state.hpp"
#include "../tet_with_sites.hpp"
#include "./tet_face_recovery.hpp"
#include "./tet_face_status.hpp"
#include "./tet_plane_normal.hpp"
#include "./tet_projected_area2.hpp"
#include "./tet_triangle_in_facet.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::topology::cdt::dt3 {

/// Whether every face `recovery` states recovered stands tiled by the lane
/// facets it owns, read off the published complex `owner` alone: each in the
/// lane, on the face's plane and in the closed face
/// (@ref tet_triangle_in_facet), their areas
/// the face's, exactly on the lattice, every site of a face read at the
/// owner's deepest depth. The face half of the phase law's oracle.
template <typename Owner, typename Faces>
auto check_tet_face_fixpoint(
    const Owner &owner, const Faces &faces,
    const tet_face_recovery<typename Owner::index_type,
                            typename Owner::coord_type,
                            typename Owner::int_type> &recovery) -> bool {
  using Index = typename Owner::index_type;
  const auto &faces_status = recovery.status;
  std::vector<std::array<Index, 3>> lane;
  state_tet_protected_facets(owner, lane);
  std::vector<std::pair<Index, std::array<Index, 3>>> owned;
  const std::size_t rows = std::min(std::size_t(owner._corners.size()),
                                    owner._protected_slots.size());
  for (std::size_t cell = 0; cell < rows; ++cell) {
    if (owner._states[cell] == tet_state::dead || !owner._protected_slots[cell])
      continue;
    for (std::size_t slot = 0; slot < 4; ++slot) {
      if (!tet_facet_is_protected(owner, Index(cell), slot) ||
          std::size_t(owner._neighbors[cell][slot]) < cell)
        continue;
      const Index who = owner._owners[4 * cell + slot];
      if (who == Owner::none)
        continue;
      const auto key = tet_facet_key(owner._corners[cell], slot);
      if (who != Owner::shared_owner) {
        owned.push_back({who, key});
        continue;
      }
      const auto stack = tet_shared_owners(owner, key);
      for (auto at = stack.first; at != stack.second; ++at)
        owned.push_back({at->second, key});
    }
  }
  std::sort(owned.begin(), owned.end());
  const auto tiles = [&owned, &lane](const auto &sites, const auto &facet,
                                     Index f) {
    using Int = typename std::decay_t<decltype(sites)>::int_type;
    using T2 = typename tf::exact::meta<Int>::T2;
    const auto &first = facet[0];
    const auto &a = sites[std::size_t(first[0])].pt;
    const auto &b = sites[std::size_t(first[1])].pt;
    const auto &c = sites[std::size_t(first[2])].pt;
    const auto axes =
        tf::exact::projection_axes<Int>(tet_plane_normal<Int>(a, b, c));
    T2 expected(0), covered(0);
    for (const auto &triangle : facet)
      expected = expected + tet_projected_area2(sites, axes, Index(triangle[0]),
                                                Index(triangle[1]),
                                                Index(triangle[2]));
    const auto held = std::equal_range(
        owned.begin(), owned.end(), std::make_pair(f, std::array<Index, 3>{}),
        [](const auto &x, const auto &y) { return x.first < y.first; });
    for (auto at = held.first; at != held.second; ++at) {
      const auto &key = at->second;
      if (!std::binary_search(lane.begin(), lane.end(), key))
        return false;
      for (auto site : key)
        if (tet_orient3d_sign<Int>(a, b, c, sites[std::size_t(site)].pt) != 0)
          return false;
      if (!tet_triangle_in_facet(sites, facet, axes, key))
        return false;
      covered =
          covered + tet_projected_area2(sites, axes, key[0], key[1], key[2]);
    }
    return covered == expected;
  };
  for (std::size_t f = 0; f < faces.size(); ++f) {
    if (faces_status[f] != tet_face_status::recovered)
      continue;
    const auto &facet = faces[f];
    if (!tet_with_sites(owner, tet_deepest_depth(owner),
                        [&tiles, &facet, f](const auto &sites) {
                          return tiles(sites, facet, Index(f));
                        }))
      return false;
  }
  return true;
}

} // namespace tf::topology::cdt::dt3
