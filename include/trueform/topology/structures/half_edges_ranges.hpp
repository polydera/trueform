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

#include "../../core/range.hpp"
#include "../half_edge.hpp"
#include "../half_edge_handle.hpp"

namespace tf::topology {

template <typename Index> struct half_edges_ranges {
  using index_type = Index;
  using half_edge_handle_t = tf::half_edge_handle<Index>;
  using edge_handle_t = tf::edge_handle<Index>;
  using vertex_handle_t = tf::vertex_handle<Index>;
  using face_handle_t = tf::face_handle<Index>;

  using he_range_t = tf::range<const tf::half_edge<Index> *, tf::dynamic_size>;
  using heh_range_t = tf::range<const half_edge_handle_t *, tf::dynamic_size>;
  using char_range_t = tf::range<const char *, tf::dynamic_size>;

  half_edges_ranges(he_range_t hes, heh_range_t fhes, heh_range_t vhes,
                    char_range_t bverts, char_range_t nmverts, Index nf,
                    Index nv)
      : _half_edges{hes}, _face_half_edges{fhes}, _vertex_half_edges{vhes},
        _boundary_vertices{bverts}, _non_manifold_vertices{nmverts},
        _n_faces{nf}, _n_vertices{nv} {}

  auto half_edges_data() const { return _half_edges; }
  auto face_half_edges() const { return _face_half_edges; }
  auto vertex_half_edges() const { return _vertex_half_edges; }
  auto boundary_vertex_data() const { return _boundary_vertices; }
  auto non_manifold_vertex_data() const { return _non_manifold_vertices; }
  auto n_faces() const -> Index { return _n_faces; }
  auto n_vertices() const -> Index { return _n_vertices; }

protected:
  he_range_t _half_edges;
  heh_range_t _face_half_edges;
  heh_range_t _vertex_half_edges;
  char_range_t _boundary_vertices;
  char_range_t _non_manifold_vertices;
  Index _n_faces;
  Index _n_vertices;
};

} // namespace tf::topology
