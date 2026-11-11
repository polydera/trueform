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

#include <cstdint>
#include <tuple>

namespace tf::intersect {

/// A self pair's claim that `vertex` of form `tag` is shared with the
/// face it met, standing at face `object`.
///
/// A PER-CORNER fact: what the vertex's face fan then states
/// (@ref tf::intersect::state_shared_vertex_fans) is a function of this
/// claim alone, so every pair of a fan's product and every record
/// standing in the same fan makes the SAME claim and states one fact.
/// The pair's own contact at the vertex is a different fact with its own
/// carrier (@ref tf::intersect::tagged_intersection).
template <typename Index> struct shared_vertex_claim {
  std::int16_t tag;
  Index object;
  Index vertex;

  friend auto operator<(const shared_vertex_claim &a,
                        const shared_vertex_claim &b) -> bool {
    return std::make_tuple(a.tag, a.object, a.vertex) <
           std::make_tuple(b.tag, b.object, b.vertex);
  }

  friend auto operator==(const shared_vertex_claim &a,
                         const shared_vertex_claim &b) -> bool {
    return std::make_tuple(a.tag, a.object, a.vertex) ==
           std::make_tuple(b.tag, b.object, b.vertex);
  }
};

} // namespace tf::intersect
