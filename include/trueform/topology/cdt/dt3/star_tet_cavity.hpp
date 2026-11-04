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
#include "./write_tet_cavity.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Construct the cavity star in the cavity's own rows first, growing the
/// lanes for the rest. Returns a star cell, used as the next location hint.
template <typename Owner>
auto star_tet_cavity(Owner &owner, typename Owner::index_type apex)
    -> typename Owner::index_type {
  using Index = typename Owner::index_type;
  const Index first = Index(owner._corners.size());
  const std::size_t walls = owner._boundary.size();
  const std::size_t reused = std::min(walls, owner._cavity.size());
  owner._corners.reallocate(std::size_t(first) + walls - reused);
  owner._neighbors.reallocate(std::size_t(first) + walls - reused);
  owner._states.reallocate(std::size_t(first) + walls - reused);
  auto &rows = owner._rows;
  rows.allocate(walls);
  for (std::size_t i = 0; i < reused; ++i)
    rows[i] = owner._cavity[i];
  for (std::size_t i = reused; i < walls; ++i)
    rows[i] = Index(std::size_t(first) + i - reused);
  write_tet_cavity(owner, apex, rows, owner._boundary, owner._links);
  owner._n_dead -= reused;
  return rows[0];
}

} // namespace tf::topology::cdt::dt3
