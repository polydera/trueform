/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./intersection_target.hpp"
#include <tuple>

namespace tf::intersect {
template <typename Index> struct intersection {
  Index mesh;
  Index polygon;
  Index polygon_other;
  intersection_target<Index> target;
  intersection_target<Index> target_other;
  Index id;

  auto polygon_key() const { return std::make_pair(mesh, polygon); }

  friend auto operator<(const intersection &i0, const intersection &i1)
      -> bool {
    return std::make_tuple(i0.mesh, i0.polygon, i0.polygon_other, i0.target,
                           i0.target_other, i0.id) <
           std::make_tuple(i1.mesh, i1.polygon, i1.polygon_other, i1.target,
                           i1.target_other, i1.id);
  }

  friend auto operator==(const intersection &i0, const intersection &i1)
      -> bool {
    return std::make_tuple(i0.mesh, i0.polygon, i0.polygon_other, i0.target,
                           i0.target_other, i0.id) ==
           std::make_tuple(i1.mesh, i1.polygon, i1.polygon_other, i1.target,
                           i1.target_other, i1.id);
  }
};
} // namespace tf::intersect
