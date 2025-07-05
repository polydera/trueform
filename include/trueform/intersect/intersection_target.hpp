/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../topology/topo_type.hpp"
#include <utility>
namespace tf::intersect {

template <typename Index> struct intersection_target {
  Index id;            // Index within the mesh
  tf::topo_type label; // Vertex, edge, face

  friend auto operator<(const intersection_target &i0,
                        const intersection_target &i1) -> bool {
    return std::make_pair(i0.label, i0.id) < std::make_pair(i1.label, i1.id);
  }

  friend auto operator==(const intersection_target &i0,
                         const intersection_target &i1) -> bool {
    return std::make_pair(i0.label, i0.id) == std::make_pair(i1.label, i1.id);
  }
};
} // namespace tf::intersect
