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
#include "./intersection_target.hpp"
#include <tuple>

namespace tf::intersect {
template <typename Index> struct tagged_intersection {
  Index tag;
  Index object;
  Index object_other;
  intersection_target<Index> target;
  intersection_target<Index> target_other;
  Index id;

  auto object_key() const { return std::make_pair(tag, object); }

  friend auto operator<(const tagged_intersection &i0,
                        const tagged_intersection &i1) -> bool {
    return std::make_tuple(i0.tag, i0.object, i0.object_other, i0.target,
                           i0.target_other, i0.id) <
           std::make_tuple(i1.tag, i1.object, i1.object_other, i1.target,
                           i1.target_other, i1.id);
  }

  friend auto operator==(const tagged_intersection &i0,
                         const tagged_intersection &i1) -> bool {
    return std::make_tuple(i0.tag, i0.object, i0.object_other, i0.target,
                           i0.target_other, i0.id) ==
           std::make_tuple(i1.tag, i1.object, i1.object_other, i1.target,
                           i1.target_other, i1.id);
  }
};
} // namespace tf::intersect
