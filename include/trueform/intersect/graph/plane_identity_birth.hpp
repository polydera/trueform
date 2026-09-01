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

#include "../../exact/dyadic_blend.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/vertex.hpp"
#include "../../exact/wide_dyadic_ratio.hpp"
#include "./plane_identity_names.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// BIRTH — the one rounding of one name, on the smallest root of its
/// run, in that root's own frame.
///
/// Both edges are LATTICE segments here: their endpoints were born at
/// the previous tier, so the crossing parameter is one 2x2 determinant
/// ratio of coordinate differences — degree two, exactly representable,
/// and the rational never leaves this expression. That is why widths
/// are depth independent: a born operand re-enters as a lattice entry
/// however deep the cascade.
template <typename Index, typename Int, typename Graph, typename Frames,
          typename GetPoint>
auto birth_plane_point(const Graph &g, const Frames &frames,
                       const GetPoint &get_point,
                       const plane_name_statement<Index> &statement) ->
    typename tf::exact::meta<Int>::param_type {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto &root = g.canon_group(statement.root)[0];
  const auto &partner = g.canon_group(statement.partner)[0];
  const auto lo = get_point(root.point_tag_0, root.point_0);
  const auto hi = get_point(root.point_tag_1, root.point_1);
  const auto p0 = get_point(partner.point_tag_0, partner.point_0);
  const auto p1 = get_point(partner.point_tag_1, partner.point_1);
  const auto &frame = frames.frame(statement.plane);
  const int ax0 = frame.ax0, ax1 = frame.ax1;
  const T1 da[2] = {T1(hi[ax0]) - lo[ax0], T1(hi[ax1]) - lo[ax1]};
  const T1 db[2] = {T1(p1[ax0]) - p0[ax0], T1(p1[ax1]) - p0[ax1]};
  const T1 w[2] = {T1(p0[ax0]) - lo[ax0], T1(p0[ax1]) - lo[ax1]};
  T2 den = T2(da[0]) * db[1] - T2(da[1]) * db[0];
  T2 num = T2(w[0]) * db[1] - T2(w[1]) * db[0];
  if (den < T2(0)) {
    den = -den;
    num = -num;
  }
  return tf::exact::wide_dyadic_ratio<Int>(num, den);
}

template <typename Index, typename Int, typename Graph, typename GetPoint>
auto birth_plane_point(const Graph &g, const GetPoint &get_point,
                       const plane_name_statement<Index> &statement) ->
    typename tf::exact::meta<Int>::param_type {
  return birth_plane_point<Index, Int>(g, g, get_point, statement);
}

/// THE one way a coordinate is read: a class states (root, dyadic t)
/// and the position is blended from the root's endpoints on demand.
/// Blending is per coordinate and a projection selects coordinates, so
/// this IS what a stored table would have held.
template <typename Index, typename Int, typename Graph, typename GetPoint,
          typename GetMeshPoint, typename VertexOffsets>
auto plane_class_position(const Graph &g, Index root,
                          typename tf::exact::meta<Int>::param_type parameter,
                          const plane_name<Index> &name,
                          const VertexOffsets &vertex_offsets,
                          const GetPoint &get_point,
                          const GetMeshPoint &get_mesh_point)
    -> tf::exact::pt3<Int> {
  if (root == Index(-1)) {
    if (name[0] == Index(0))
      return get_point(std::int16_t(-1), name[1]);
    const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, name[1]);
    return get_mesh_point(int(tag), name[1] - vertex_offsets[std::size_t(tag)]);
  }
  const auto &def = g.canon_group(root)[0];
  const auto lo = get_point(def.point_tag_0, def.point_0);
  const auto hi = get_point(def.point_tag_1, def.point_1);
  tf::exact::pt3<Int> point;
  for (int coordinate = 0; coordinate < 3; ++coordinate)
    point[coordinate] =
        tf::exact::dyadic_blend(lo[coordinate], hi[coordinate], parameter);
  return point;
}

} // namespace tf::intersect::graph
