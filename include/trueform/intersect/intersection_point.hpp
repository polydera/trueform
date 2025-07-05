/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
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
