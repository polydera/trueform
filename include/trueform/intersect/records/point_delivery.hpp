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

#include "../../topology/topo_id.hpp"
#include <cstdint>
#include <tuple>

namespace tf::intersect {

/// One face's claim on a canonical point: the point lies at `target` of
/// face `object` of form `tag`.
///
/// A PER-FACE fact, so a contact fanned across a feature's whole face fan
/// costs the SUM of the two fans. The pair it was proved on is a
/// different fact with its own carrier
/// (@ref tf::intersect::tagged_intersection).
///
/// Ordering puts the point id ahead of the target, so one face's
/// deliveries ascend BY POINT — the sorted set the fan's gate merges.
template <typename Index> struct point_delivery {
  std::int16_t tag;
  Index object;
  tf::topo_id<Index> target;
  Index id;

  auto key() const { return std::make_tuple(tag, object); }

  friend auto operator<(const point_delivery &a, const point_delivery &b)
      -> bool {
    return std::make_tuple(a.tag, a.object, a.id, a.target) <
           std::make_tuple(b.tag, b.object, b.id, b.target);
  }

  friend auto operator==(const point_delivery &a, const point_delivery &b)
      -> bool {
    return std::make_tuple(a.tag, a.object, a.id, a.target) ==
           std::make_tuple(b.tag, b.object, b.id, b.target);
  }
};

} // namespace tf::intersect
