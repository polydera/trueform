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
#include "./point_source.hpp"
#include <utility>

namespace tf::intersect {
template <typename Index> struct intersection_point {
  Index id;
  point_source label;

  friend auto operator==(const intersection_point &i0,
                         const intersection_point &i1) {
    return std::make_pair(i0.label, i0.id) == std::make_pair(i1.label, i1.id);
  }
  friend auto operator!=(const intersection_point &i0,
                         const intersection_point &i1) {
    return std::make_pair(i0.label, i0.id) != std::make_pair(i1.label, i1.id);
  }
  friend auto operator<(const intersection_point &i0,
                        const intersection_point &i1) {
    return std::make_pair(i0.label, i0.id) < std::make_pair(i1.label, i1.id);
  }
};
} // namespace tf::intersect
