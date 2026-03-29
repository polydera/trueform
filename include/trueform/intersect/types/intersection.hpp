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
#include "../../topology/topo_id.hpp"
#include <tuple>

namespace tf::intersect {
template <typename Index> struct intersection {
  Index object;
  Index object_other;
  tf::topo_id<Index> target;
  tf::topo_id<Index> target_other;
  Index id;

  auto object_key() const { return object; }

  friend auto operator<(const intersection &i0, const intersection &i1)
      -> bool {
    return std::make_tuple(i0.object, i0.object_other, i0.target,
                           i0.target_other, i0.id) <
           std::make_tuple(i1.object, i1.object_other, i1.target,
                           i1.target_other, i1.id);
  }

  friend auto operator==(const intersection &i0, const intersection &i1)
      -> bool {
    return std::make_tuple(i0.object, i0.object_other, i0.target,
                           i0.target_other, i0.id) ==
           std::make_tuple(i1.object, i1.object_other, i1.target,
                           i1.target_other, i1.id);
  }
};
template <typename Index>
auto make_canonical_intersection(Index object, Index object_other,
                                 tf::topo_id<Index> target,
                                 tf::topo_id<Index> target_other, Index id) {
  return intersection<Index>{object, object_other, target, target_other, id};
}
} // namespace tf::intersect
