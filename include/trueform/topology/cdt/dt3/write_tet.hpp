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
#include "../../../core/config/verify.hpp"
#include "./tet_state.hpp"
#include "./tet_round_state.hpp"
#include "./write_tet_topology.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Write and publish one serial cell. Returns the final slot of c3.
template <typename Owner>
auto write_tet(Owner &owner, typename Owner::index_type tet,
               typename Owner::index_type c0, typename Owner::index_type c1,
               typename Owner::index_type c2, typename Owner::index_type c3,
               tet_round_state<typename Owner::index_type> &round)
    -> std::size_t {
  if constexpr (tf::core::verification)
    tf::core::verify(!round.open() || (c0 >= 0 &&
                                       std::size_t(c0) < round.sites.size() &&
                                       round.sites[std::size_t(c0)] ==
                                           round.patch),
                     "a region writes only rows its own patch names");
  const auto base = write_tet_topology(owner, tet, c0, c1, c2, c3, round);
  owner._states[std::size_t(tet)] =
      owner._corners[std::size_t(tet)][3] == Owner::infinite
          ? tet_state::infinite
          : tet_state::live;
  return base;
}

/// The same outside a round.
template <typename Owner>
auto write_tet(Owner &owner, typename Owner::index_type tet,
               typename Owner::index_type c0, typename Owner::index_type c1,
               typename Owner::index_type c2, typename Owner::index_type c3)
    -> std::size_t {
  tet_round_state<typename Owner::index_type> closed;
  return write_tet(owner, tet, c0, c1, c2, c3, closed);
}

} // namespace tf::topology::cdt::dt3
