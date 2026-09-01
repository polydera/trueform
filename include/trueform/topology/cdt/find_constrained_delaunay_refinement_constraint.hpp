/*
 * Copyright (c) 2025 XLAB
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
#include <tuple>

namespace tf::topology::cdt {

template <typename Owner>
auto find_constrained_delaunay_refinement_constraint(
    const Owner &owner, typename Owner::index_type u,
    typename Owner::index_type v, bool broadcast)
    -> typename Owner::index_type {
  using Index = typename Owner::index_type;

  const std::array<Index, 2> key{std::min(u, v), std::max(u, v)};
  const auto it = std::lower_bound(
      owner._constraint_aliases.begin(), owner._constraint_aliases.end(), key,
      [](const auto &span, const auto &value) {
        return std::tie(span.first, span.second) <
               std::tie(value[0], value[1]);
      });
  if (it != owner._constraint_aliases.end() && it->first == key[0] &&
      it->second == key[1])
    return broadcast
               ? Index(it - owner._constraint_aliases.begin())
               : it->input;
  return Owner::none;
}

} // namespace tf::topology::cdt
