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

template <typename Index> struct tagged_intersection {
  Index tag;
  Index tag_other;
  Index object;
  Index object_other;
  tf::topo_id<Index> target;
  tf::topo_id<Index> target_other;
  Index id;

  auto key() const { return std::make_tuple(tag, object); }

  friend auto operator<(const tagged_intersection &a,
                        const tagged_intersection &b) -> bool {
    return std::make_tuple(a.tag, a.object, a.tag_other, a.object_other,
                           a.target, a.target_other, a.id) <
           std::make_tuple(b.tag, b.object, b.tag_other, b.object_other,
                           b.target, b.target_other, b.id);
  }

  friend auto operator==(const tagged_intersection &a,
                         const tagged_intersection &b) -> bool {
    return std::make_tuple(a.tag, a.object, a.tag_other, a.object_other,
                           a.target, a.target_other, a.id) ==
           std::make_tuple(b.tag, b.object, b.tag_other, b.object_other,
                           b.target, b.target_other, b.id);
  }
};

} // namespace tf::intersect
