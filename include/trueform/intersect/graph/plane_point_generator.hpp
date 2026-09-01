/*
 * Copyright (c) 2026 XLAB
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

#include "../../exact/edge_parameter.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/vertex.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// A canonical intersection point as its GENERATORS state it: an
/// original vertex, or an exact fraction of an original edge's span.
/// The materialized position is never read here — rounding it is the
/// error the carried orientation exists to remove.
template <typename Index, typename Int> struct plane_point_generator {
  tf::exact::pt3<Int> u{};
  tf::exact::pt3<Int> v{};
  tf::exact::edge_parameter<Int> t{};
  Index carrier_u = Index(-1);
  Index carrier_v = Index(-1);
  bool on_edge = false;
};

template <typename Index, typename Int, typename Ibp, typename VertexOffsets,
          typename GetPoint>
auto plane_point_generator_of(const Ibp &ibp,
                              const VertexOffsets &vertex_offsets,
                              const GetPoint &get_point, Index id)
    -> plane_point_generator<Index, Int> {
  plane_point_generator<Index, Int> generator;
  if (id < ibp.n_vertex_points()) {
    const auto &anchor = ibp.vertex_anchor(id);
    generator.u = get_point(anchor.tag, anchor.vid);
    return generator;
  }
  auto flat_point = [&](Index flat) {
    const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
    return get_point(std::int16_t(tag),
                     flat - vertex_offsets[std::size_t(tag)]);
  };
  const auto &carrier = ibp.home_edge(id);
  generator.u = flat_point(carrier.u);
  generator.v = flat_point(carrier.v);
  generator.t = ibp.exact_parameter(id);
  generator.carrier_u = carrier.u;
  generator.carrier_v = carrier.v;
  generator.on_edge = true;
  return generator;
}

} // namespace tf::intersect::graph
