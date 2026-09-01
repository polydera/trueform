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
#include "../../core/point.hpp"
#include "./queue_constrained_delaunay_refinement_triangle.hpp"
#include "./split_constrained_delaunay_refinement_edge.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt {

template <typename Owner>
auto drain_constrained_delaunay_refinement_generation(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;

  owner._generation.clear();
  std::swap(owner._generation, owner._pending);
  for (const auto &request : owner._generation) {
    Index face = request.f;
    int edge = int(request.e);
    if (owner._t[face].stamp != request.stamp) {
      queue_constrained_delaunay_refinement_triangle(owner, face);
      continue;
    }
    if (!owner.constrained(face, edge))
      continue;
    Index u = owner._t[face].v[edge];
    Index v = owner._t[face].v[(edge + 1) % 3];
    Index point = Index(owner._ip.size());
    owner._ip.push_back(tf::point<Int, 2>{
        Int((std::int64_t(owner._ip[u][0]) + owner._ip[v][0]) / 2),
        Int((std::int64_t(owner._ip[u][1]) + owner._ip[v][1]) / 2)});
    if ((owner._ip[point][0] == owner._ip[u][0] &&
         owner._ip[point][1] == owner._ip[u][1]) ||
        (owner._ip[point][0] == owner._ip[v][0] &&
         owner._ip[point][1] == owner._ip[v][1])) {
      owner._ip.pop_back();
      continue;
    }
    owner._dp.push_back(
        {(double(owner._ip[point][0]) - owner._offx) / owner._scale,
         (double(owner._ip[point][1]) - owner._offy) / owner._scale});
    Index segment = owner._t[face].seg[edge];
    Index segment_lo = Owner::none, segment_hi = Owner::none;
    if (segment != Owner::none) {
      auto value = owner._segments[std::size_t(segment)];
      owner._splits.push_back(
          {value.origin, std::uint8_t(2 * value.m + 1),
           std::uint8_t(value.d + 1)});
      segment_lo = Index(owner._segments.size());
      segment_hi = segment_lo + 1;
      owner._segments.push_back(
          {value.origin, value.lo_vertex, std::uint8_t(2 * value.m),
           std::uint8_t(value.d + 1)});
      owner._segments.push_back(
          {value.origin, point, std::uint8_t(2 * value.m + 1),
           std::uint8_t(value.d + 1)});
    }
    owner._con_nbr.push_back({u, v});
    auto relink = [&](Index x, Index from, Index to) {
      if (owner._con_nbr[std::size_t(x)][0] == from)
        owner._con_nbr[std::size_t(x)][0] = to;
      else if (owner._con_nbr[std::size_t(x)][1] == from)
        owner._con_nbr[std::size_t(x)][1] = to;
    };
    relink(u, v, point);
    relink(v, u, point);
    split_constrained_delaunay_refinement_edge(
        owner, face, edge, point, segment_lo, segment_hi);
  }
  for (const auto &request : owner._pending)
    queue_constrained_delaunay_refinement_triangle(owner, request.f);
  owner._pending.clear();
}

} // namespace tf::topology::cdt
