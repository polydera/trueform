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
#include "../../../core/small_vector.hpp"
#include "./locate_conflict_tet.hpp"
#include "./tet_site_position.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <unordered_set>

namespace tf::topology::cdt::dt3 {

/// The facet ticket `4 * cell + slot` of a live cell holding the facet
/// `key`, corner slots ascending, or `none` when the complex holds no such
/// facet: the cells around the key's first corner, entered from its site
/// cell while one is recorded, are read until one holds the other two. The
/// cells read are a per-query traversal set, scanned while small and hashed
/// once a star outgrows the scan.
template <typename Owner>
auto locate_tet_facet(const Owner &owner,
                      const std::array<typename Owner::index_type, 3> &key) ->
    typename Owner::index_type {
  using Index = typename Owner::index_type;
  const Index a = key[0];
  if (a < 0 || std::size_t(a) >= tet_site_count(owner) ||
      !owner._corners.size())
    return Owner::none;
  const auto holds = [&owner, a](Index cell) {
    if (cell < 0 || std::size_t(cell) >= std::size_t(owner._corners.size()) ||
        owner._states[std::size_t(cell)] == tet_state::dead)
      return false;
    for (auto corner : owner._corners[std::size_t(cell)])
      if (corner == a)
        return true;
    return false;
  };
  Index from = Owner::none;
  if (owner._record_site_cells && std::size_t(a) < owner._site_cells.size())
    from = owner._site_cells.load(std::size_t(a));
  if (!holds(from))
    from = locate_conflict_tet(owner, a);
  if (!holds(from))
    return Owner::none;
  tf::small_vector<Index, 64> star;
  std::unordered_set<Index> seen;
  const auto admit = [&star, &seen](Index cell) {
    if (star.size() < 32) {
      if (std::find(star.begin(), star.end(), cell) != star.end())
        return;
    } else {
      if (!seen.size())
        seen.insert(star.begin(), star.end());
      if (!seen.insert(cell).second)
        return;
    }
    star.push_back(cell);
  };
  star.push_back(from);
  for (std::size_t head = 0; head < star.size(); ++head) {
    const Index cell = star[head];
    const auto corners = owner._corners[std::size_t(cell)];
    std::size_t found = 0, other = 4;
    for (std::size_t j = 0; j < 4; ++j) {
      if (corners[j] == a || corners[j] == key[1] || corners[j] == key[2])
        ++found;
      else
        other = j;
    }
    if (found == 3 && other < 4)
      return Index(4 * std::size_t(cell) + other);
    for (std::size_t j = 0; j < 4; ++j) {
      if (corners[j] == a)
        continue;
      const Index next = owner._neighbors[std::size_t(cell)][j];
      if (holds(next))
        admit(next);
    }
  }
  return Owner::none;
}

} // namespace tf::topology::cdt::dt3
