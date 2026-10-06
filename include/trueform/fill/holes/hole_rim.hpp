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
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/vector.hpp"
#include "../../exact/vertex.hpp"
#include "../hole_split.hpp"
#include "./hole_lattice.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::fill {

/// The rim edge leaving one position of a prepared cycle.
///
/// The carrying triangle is the original mesh face the edge's ORIGINAL edge
/// belongs to, which every subedge of a split inherits; `forward` says that
/// face winds the original edge the way the cycle runs it, which is what
/// turns the face's own plane into the `normal` a patch is measured against
/// — already turned to the patch's side, and already in the type the metric
/// is decided in. `t0` and `t1` are the subedge's ends as parameters of the
/// original edge in its canonical direction, so a position is an original
/// vertex exactly when its `t0` is an end of that scale.
///
/// @tparam Index The mesh's index type.
template <typename Index> struct hole_rim_edge {
  Index face;
  Index v0;
  Index v1;
  tf::vector<double, 3> normal;
  std::uint8_t t0;
  std::uint8_t t1;
  bool forward;
};

/// One rim prepared for filling: its cycle rooted and turned the way the
/// elected ticket states, on the scaled lattice, in the mesh's own
/// coordinates, and named in the flat identity space the patch emits.
///
/// Position `k`'s outgoing edge is `edges[k]`, the cycle closing from the
/// last position back to the first.
///
/// @tparam Index The mesh's index type.
/// @tparam Int The lattice type the predicates dispatch on.
/// @tparam RealT The mesh's coordinate type.
template <typename Index, typename Int, typename RealT> struct hole_rim {
  tf::buffer<tf::exact::vertex<Index, tf::fill::hole_site_coord<Int>>> sites;
  tf::buffer<tf::point<RealT, 3>> positions;
  tf::buffer<Index> corners;
  tf::buffer<tf::fill::hole_rim_edge<Index>> edges;

  auto size() const -> std::size_t { return corners.size(); }

  auto clear() -> void {
    sites.clear();
    positions.clear();
    corners.clear();
    edges.clear();
  }
};

/// Whether a prepared position is a split of an original edge rather than an
/// original vertex, which its outgoing subedge's own parameter states.
template <typename Index, typename Int, typename RealT>
auto hole_position_is_split(const tf::fill::hole_rim<Index, Int, RealT> &rim,
                            Index position) -> bool {
  const auto parameter = rim.edges[std::size_t(position)].t0;
  return parameter != 0 && parameter != tf::hole_split_scale;
}

} // namespace tf::fill
