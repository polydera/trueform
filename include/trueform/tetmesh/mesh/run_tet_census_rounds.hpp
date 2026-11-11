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
#include <cstddef>

namespace tf::tetmesh {

/// What one census round did: the sites it collapsed, the faces it moved and
/// recovered again, and whether the recovery held its capacity.
struct tet_census_round {
  std::size_t collapsed = 0;
  std::size_t moved = 0;
  bool recovered = true;
};

/// The census's rounds: `round()` convicts, collapses and recovers the faces
/// it moved, until a round collapses nothing, at most `cap` rounds. A census
/// the cap cuts short counts the faces its last round moved into
/// `collapse_refused`, so exhaustion is never silent. Returns false where a
/// round's recovery ran out of capacity.
template <typename Round>
auto run_tet_census_rounds(std::size_t cap, const Round &round,
                           std::size_t &collapse_refused) -> bool {
  for (std::size_t at = 0; at < cap; ++at) {
    const tet_census_round made = round();
    if (!made.recovered)
      return false;
    if (!made.collapsed)
      return true;
    if (at + 1 == cap)
      collapse_refused += made.moved;
  }
  return true;
}

} // namespace tf::tetmesh
