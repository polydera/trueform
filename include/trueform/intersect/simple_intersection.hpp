/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./intersection_target.hpp"
#include <tuple>

namespace tf::intersect {

template <typename Index> struct simple_intersection {
  Index object;
  intersection_target<Index> target;
  Index id;

  auto object_key() const { return object; }

  friend auto operator<(const simple_intersection &intersection0,
                        const simple_intersection &intersection1) -> bool {
    return std::make_tuple(intersection0.object, intersection0.target.label,
                           intersection0.target.id,
                           intersection0.id) <
           std::make_tuple(intersection1.object, intersection1.target.label,
                           intersection1.target.id,
                           intersection1.id);
  }

  friend auto operator==(const simple_intersection &intersection0,
                         const simple_intersection &intersection1) -> bool {
    return std::make_tuple(intersection0.object, intersection0.target.label,
                           intersection0.target.id,
                           intersection0.id) ==
           std::make_tuple(intersection1.object, intersection1.target.label,
                           intersection1.target.id,
                           intersection1.id);
  }
};
} // namespace tf::intersect
