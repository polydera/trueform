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

#include "../../core/buffer.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// Name the one input constraint a claimed edge speaks for, for every input
/// that claimed it.
///
/// Inputs are recovered in ascending order and each claim overwrites the
/// edge's provenance, so the standing owner of a coincidence class is its
/// LAST member. A producer that names a different member of the same class —
/// a refinement states its alias representative — resolves through this map
/// and reads the same attribution.
template <typename Index, typename Collisions>
auto union_plane_constraint_collisions(const Collisions &collisions,
                                       Index n_constraints,
                                       tf::buffer<Index> &roots) -> void {
  roots.allocate(std::size_t(n_constraints));
  for (Index input = 0; input < n_constraints; ++input)
    roots[std::size_t(input)] = input;
  const auto find = [&roots](Index input) {
    while (roots[std::size_t(input)] != input)
      input = roots[std::size_t(input)] =
          roots[std::size_t(roots[std::size_t(input)])];
    return input;
  };
  for (const auto &collision : collisions) {
    if (collision.prior_input < Index(0) || collision.input < Index(0) ||
        n_constraints <= collision.prior_input ||
        n_constraints <= collision.input)
      continue;
    const auto prior = find(collision.prior_input);
    const auto claim = find(collision.input);
    if (prior != claim)
      roots[std::size_t(std::min(prior, claim))] = std::max(prior, claim);
  }
  for (Index input = 0; input < n_constraints; ++input)
    roots[std::size_t(input)] = find(input);
}

} // namespace tf::arrangement
