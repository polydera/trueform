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
#include "../cdt_region_mode.hpp"
#include "./constrained_delaunay_face_edges.hpp"
#include "./materialize_constrained_delaunay_face_adjacency.hpp"
#include "./size_adaptive.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto materialize_constrained_delaunay_regions(Owner &owner,
                                              tf::cdt_region_mode mode)
    -> void {
  using Index = typename Owner::index_type;
  materialize_constrained_delaunay_face_adjacency(owner);
  owner._region_labels.clear();
  owner._region_mode = mode;

  const auto &first_edges = owner._first_edge_of_triangle;
  if (first_edges.size() == 0)
    return;

  const std::size_t n_darts = owner._edges.size();
  const auto &triangle_of_edge = owner._triangle_of_edge;
  auto &labels = owner._region_labels;
  labels.allocate(first_edges.size());
  topology::cdt::fill_auto(labels, Owner::none);

  const auto is_wall = [&](Index edge) {
    return owner._edges[std::size_t(edge)].constrained ==
           Owner::boundary_constrained;
  };
  const bool stop_at_walls = mode == tf::cdt_region_mode::components;

  auto &stack = owner._scratch_stack;
  stack.clear();

  const auto flood = [&] {
    while (stack.size() != 0) {
      const Index triangle = stack.back();
      stack.pop_back();
      const Index label = labels[std::size_t(triangle)];

      for (const Index edge :
           constrained_delaunay_face_edges(owner, triangle)) {
        if (stop_at_walls && is_wall(edge))
          continue;
        const Index neighbor = triangle_of_edge[std::size_t(edge ^ Index(1))];
        if (neighbor == Owner::none ||
            labels[std::size_t(neighbor)] != Owner::none)
          continue;
        labels[std::size_t(neighbor)] =
            label ^ (is_wall(edge) ? Index(1) : Index(0));
        stack.push_back(neighbor);
      }
    }
  };

  for (std::size_t edge = 0; edge < n_darts; ++edge) {
    const Index dart = static_cast<Index>(edge);
    if (!owner._edges[edge].boundary || (stop_at_walls && is_wall(dart)))
      continue;
    const Index triangle = triangle_of_edge[edge ^ std::size_t(1)];
    if (triangle == Owner::none || labels[std::size_t(triangle)] != Owner::none)
      continue;

    labels[std::size_t(triangle)] = is_wall(dart) ? Index(1) : Index(0);
    stack.push_back(triangle);
    if (!stop_at_walls)
      break;
  }
  flood();

  Index next_label = Index(1);
  for (Index triangle = 0; triangle < static_cast<Index>(first_edges.size());
       ++triangle) {
    if (labels[std::size_t(triangle)] != Owner::none)
      continue;
    labels[std::size_t(triangle)] = stop_at_walls ? next_label++ : Index(0);
    stack.push_back(triangle);
    flood();
  }
}

} // namespace tf::topology::cdt
