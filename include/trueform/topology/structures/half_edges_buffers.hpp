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

#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../half_edge.hpp"
#include "../half_edge_handle.hpp"

namespace tf::topology {

template <typename Index> struct half_edges_buffers {
  using index_type = Index;
  using half_edge_handle_t = tf::half_edge_handle<Index>;
  using edge_handle_t = tf::edge_handle<Index>;
  using vertex_handle_t = tf::vertex_handle<Index>;
  using face_handle_t = tf::face_handle<Index>;

  half_edges_buffers() = default;

  auto half_edges_data() const { return tf::make_range(_half_edges); }
  auto half_edges_data() { return tf::make_range(_half_edges); }
  auto face_half_edges() const { return tf::make_range(_face_half_edges); }
  auto face_half_edges() { return tf::make_range(_face_half_edges); }
  auto vertex_half_edges() const { return tf::make_range(_vertex_half_edges); }
  auto vertex_half_edges() { return tf::make_range(_vertex_half_edges); }
  auto boundary_vertex_data() const {
    return tf::make_range(_boundary_vertices);
  }
  auto n_faces() const -> Index { return _n_faces; }
  auto n_vertices() const -> Index { return _n_vertices; }

  auto half_edges_data_buffer() -> tf::buffer<tf::half_edge<Index>> & {
    return _half_edges;
  }
  auto half_edges_data_buffer() const
      -> const tf::buffer<tf::half_edge<Index>> & {
    return _half_edges;
  }
  auto face_half_edges_buffer() -> tf::buffer<half_edge_handle_t> & {
    return _face_half_edges;
  }
  auto face_half_edges_buffer() const
      -> const tf::buffer<half_edge_handle_t> & {
    return _face_half_edges;
  }
  auto vertex_half_edges_buffer() -> tf::buffer<half_edge_handle_t> & {
    return _vertex_half_edges;
  }
  auto vertex_half_edges_buffer() const
      -> const tf::buffer<half_edge_handle_t> & {
    return _vertex_half_edges;
  }
  auto boundary_vertex_data_buffer() -> tf::buffer<char> & {
    return _boundary_vertices;
  }
  auto boundary_vertex_data_buffer() const -> const tf::buffer<char> & {
    return _boundary_vertices;
  }

protected:
  tf::buffer<tf::half_edge<Index>> _half_edges;
  tf::buffer<half_edge_handle_t> _face_half_edges;
  tf::buffer<half_edge_handle_t> _vertex_half_edges;
  tf::buffer<char> _boundary_vertices;
  Index _n_faces = 0;
  Index _n_vertices = 0;
};

} // namespace tf::topology
