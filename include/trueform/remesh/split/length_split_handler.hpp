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

#include "../../core/frame_of.hpp"
#include "../../core/point.hpp"
#include "../../core/transformed.hpp"
#include "../../topology/half_edges.hpp"

namespace tf {
namespace remesh {

template <typename Real, std::size_t Dims> struct length_split_handler {
  Real _max_length2;

  length_split_handler(Real max_length)
      : _max_length2(max_length * max_length) {}

  template <typename Index>
  auto max_length2(tf::edge_handle<Index>) const -> Real {
    return _max_length2;
  }

  template <typename Index>
  auto should_skip(tf::edge_handle<Index>) const -> bool {
    return false;
  }

  template <typename Index, typename PointsPolicy>
  auto distance2(const tf::half_edges<Index> &he,
                 const tf::points<PointsPolicy> &points,
                 tf::half_edge_handle<Index> heh) const -> Real {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    return tf::transformed(points[v1] - points[v0], tf::frame_of(points))
        .length2();
  }

  template <typename Index, typename PointsPolicy>
  auto interpolate(const tf::half_edges<Index> &he,
                   const tf::points<PointsPolicy> &points,
                   tf::half_edge_handle<Index> heh) const
      -> tf::point<Real, Dims> {
    auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
    return points[v0] +
           (points[v1].as_vector_view() - points[v0].as_vector_view()) *
               Real(0.5);
  }
};

} // namespace remesh
} // namespace tf
