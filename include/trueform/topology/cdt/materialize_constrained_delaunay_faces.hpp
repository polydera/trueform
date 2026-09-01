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
#include "./is_owned_constrained_delaunay_face.hpp"
#include "./materialize_constrained_delaunay_faces_parallel.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// The face carrier and the canonical dart that names it: one pass states
/// both, so the adjacency phase reads the dart this producer already wrote.
template <typename Owner>
auto materialize_constrained_delaunay_faces(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  const std::size_t n_darts = owner._edges.size();
  if (n_darts == 0) {
    owner._faces.clear();
    owner._first_edge_of_triangle.clear();
    return;
  }

  using ExecutionPolicy = typename Owner::execution_policy;
  if constexpr (ExecutionPolicy::parallel)
    if (materialize_constrained_delaunay_faces_parallel(owner, n_darts))
      return;

  const std::size_t maximum_faces = 2 * owner._points.size();
  auto &faces = owner._faces;
  auto &first_edges = owner._first_edge_of_triangle;
  faces.allocate(maximum_faces);
  first_edges.allocate(maximum_faces);
  auto *vertex_output = faces.data_buffer().begin();
  auto *first_output = first_edges.begin();

  for (std::size_t edge = 0; edge < n_darts; ++edge) {
    const Index first = static_cast<Index>(edge);
    Index second = Owner::none;
    Index third = Owner::none;
    if (!is_owned_constrained_delaunay_face(owner, first, second, third))
      continue;

    *first_output++ = first;
    *vertex_output++ = owner.output_vertex(owner._edges[edge].vertex);
    *vertex_output++ =
        owner.output_vertex(owner._edges[std::size_t(second)].vertex);
    *vertex_output++ =
        owner.output_vertex(owner._edges[std::size_t(third)].vertex);
  }
  faces.data_buffer().erase_till_end(vertex_output);
  first_edges.erase_till_end(first_output);
}

} // namespace tf::topology::cdt
