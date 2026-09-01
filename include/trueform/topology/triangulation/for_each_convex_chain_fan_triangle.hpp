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

namespace tf::topology {

/// CORE. THE FAN OF A CONVEX RUN of a cyclic chain: the run's first identity
/// is the apex, and every consecutive pair after it closes one triangle. The
/// run keeps the chain's own direction, so the fan turns the way the chain
/// does.
///
/// A CONVEX RUN IS THE WHOLE TRIANGULATION OF THE PIECE IT BOUNDS — no point
/// is added, no diagonal crosses the boundary, and the triangles tile it
/// exactly. That is why a state able to prove convexity never builds a
/// triangulation, and why every such state answers here rather than owning a
/// second fan of its own.
template <typename Chain, typename Emit>
auto for_each_convex_chain_fan_triangle(const Chain &chain, std::size_t start,
                                        std::size_t count, const Emit &emit)
    -> void {
  const auto n = chain.size();
  for (std::size_t at = 1; at + 1 < count; ++at)
    emit(chain[start], chain[(start + at) % n],
         chain[(start + at + 1) % n]);
}

} // namespace tf::topology
