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
#include "./plane_arrangement_census.hpp"
#include "./plane_round_evidence.hpp"
#include "./state_plane_round_frontier.hpp"

#include <cassert>
#include <cstddef>

namespace tf::arrangement {

/// WAVE. THE BARRIER a lazy world's first round reaches.
///
/// A world that has not materialized its definition tier states no canonical
/// group, and a refusal or a weld is exactly the fact that needs one: a
/// refusal's statement names a group as its root, and a weld's substitution
/// reaches every carrier of one. So the tier becomes real HERE, once, and the
/// carriers that saw something read it again.
///
/// THE EXTENT LAW: a world states its canonical extent exactly ONCE, at the
/// barrier that makes the group space real. Before this barrier nothing this
/// arrangement holds is expressed in the group space — no ticket, no router,
/// no merge, no created class — so the freeze here is the FIRST statement of
/// the extent, not a second producer of it.
///
/// The round's evidence is DISCARDED rather than translated: translating it
/// would be a second producer of the statements, and the set is the frontier
/// this returns — |refused| + |welded| carriers, re-triangulated against the
/// real tables at one extra build each.
///
/// Returns whether the barrier fired; the round is a retry when it did.
template <typename Index, typename Int, typename World>
auto close_plane_lazy_round(World &world,
                            plane_round_evidence<Index, Int> &evidence,
                            Index &immutable_canon_extent,
                            plane_arrangement_census &census,
                            tf::buffer<Index> &frontier) -> bool {
  if (world.materialized() ||
      (evidence.refused.size() == 0 && evidence.welds.size() == 0))
    return false;
  assert(immutable_canon_extent == Index(0));
  evidence.census.refusals += evidence.refused.size();
  census += evidence.census;
  census.planes = std::size_t(world.n_planes());

  state_plane_round_frontier(evidence, frontier);
  world.materialize();
  immutable_canon_extent = world.n_canon();
  return true;
}

} // namespace tf::arrangement
