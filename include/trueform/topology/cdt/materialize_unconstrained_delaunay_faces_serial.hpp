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
#include "./mark_unconstrained_delaunay_face_visited.hpp"
#include "./unconstrained_delaunay_triangle_orbit.hpp"
#include <cassert>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner, typename VisitBuffer>
auto materialize_unconstrained_delaunay_faces_serial(Owner &owner,
                                                     VisitBuffer &visited)
    -> void {
  using Index = typename Owner::index_type;
  using VertexPolicy = typename Owner::vertex_policy;
  owner._faces.allocate(owner._sites.size() * 2);
  auto *output = owner._faces.data_buffer().begin();
  for (Index first = 0; first < static_cast<Index>(owner._edges.size());
       ++first) {
    if (owner._edges[std::size_t(first)].vertex == Owner::none ||
        visited[std::size_t(first)] != 0)
      continue;
    const auto orbit = trace_unconstrained_delaunay_triangle(owner, first);
    assert(orbit.closed);
    if (!orbit.closed)
      continue;
    const Index second = orbit.darts[1];
    const Index third = orbit.darts[2];
    const Index first_vertex = owner._edges[std::size_t(first)].vertex;
    const Index second_vertex = owner._edges[std::size_t(second)].vertex;
    const Index third_vertex = owner._edges[std::size_t(third)].vertex;
    mark_unconstrained_delaunay_face_visited(visited, orbit);
    *output++ = VertexPolicy::output(first_vertex,
                                     owner._sites[std::size_t(first_vertex)]);
    *output++ = VertexPolicy::output(second_vertex,
                                     owner._sites[std::size_t(second_vertex)]);
    *output++ = VertexPolicy::output(third_vertex,
                                     owner._sites[std::size_t(third_vertex)]);
  }
  owner._faces.data_buffer().erase_till_end(output);
}

} // namespace tf::topology::cdt
