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
#include "./tet_round_state.hpp"
#include "./tet_facet_owner.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Add `face` to the stack of the shared facet `key` with the `layers` of it
/// the facet covers, once: the row joins the table in its sorted place, and
/// a row the face holds already takes the layers stated now. WHILE A ROUND
/// RUNS THE ROW IS THE REGION'S OWN (@ref tet_round_state), merged into the
/// table in region order at the barrier — the table is sorted and one
/// insertion moves it, so no two regions may hold it at once.
template <typename Owner, typename Index>
auto share_tet_facet(Owner &owner,
                     const std::array<typename Owner::index_type, 3> &key,
                     typename Owner::index_type face,
                     typename Owner::index_type layers,
                     tet_round_state<Index> &round) -> void {
  if (round.open()) {
    for (auto &pending : round.shared)
      if (pending.key == key && pending.face == face) {
        pending.layers = layers;
        return;
      }
    round.shared.push_back(
        tet_facet_owner<typename Owner::index_type>{key, face, layers});
    return;
  }
  auto &shared = owner._shared;
  const tet_facet_owner<typename Owner::index_type> row{key, face, layers};
  const auto at = std::lower_bound(
      shared.begin(), shared.end(), row, [](const auto &x, const auto &y) {
        return x.key < y.key || (x.key == y.key && x.face < y.face);
      });
  if (at != shared.end() && at->key == key && at->face == face) {
    at->layers = layers;
    return;
  }
  const auto offset = std::ptrdiff_t(at - shared.begin());
  shared.push_back(row);
  std::rotate(shared.begin() + offset, shared.end() - 1, shared.end());
}

/// The same outside a round, where the table is the one task's.
template <typename Owner>
auto share_tet_facet(Owner &owner,
                     const std::array<typename Owner::index_type, 3> &key,
                     typename Owner::index_type face,
                     typename Owner::index_type layers = 1) -> void {
  tet_round_state<typename Owner::index_type> closed;
  share_tet_facet(owner, key, face, layers, closed);
}

} // namespace tf::topology::cdt::dt3
