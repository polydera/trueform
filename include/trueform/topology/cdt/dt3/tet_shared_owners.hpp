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
#include <algorithm>
#include <array>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// The faces owning the shared facet `key` (@ref tetrahedralization_owner
/// `_shared`): the rows of its stack, as a range of the table, empty for a
/// facet no stack holds.
template <typename Owner>
auto tet_shared_owners(const Owner &owner,
                       const std::array<typename Owner::index_type, 3> &key) {
  const auto &shared = owner._shared;
  return std::equal_range(
      shared.begin(), shared.end(), key,
      [](const auto &x, const auto &y) {
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>,
                                     std::array<typename Owner::index_type, 3>>)
          return x < y.first;
        else
          return x.first < y;
      });
}

} // namespace tf::topology::cdt::dt3
