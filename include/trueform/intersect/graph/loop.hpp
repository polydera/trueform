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

#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/buffer.hpp"
#include "../../exact/int128.hpp"
#include "../../topology/topo_type.hpp"
#include "./vertex.hpp"

#include <algorithm>
#include <tuple>

namespace tf::intersect::graph {

/// Work node for parametric sorting of intersection points along an edge.
template <typename Index> struct loop_node {
  Index target_id;
  tf::topo_type label;
  Index point_id;
  Index flat_id;
  tf::exact::int128 t;
};

/// Extract a boundary loop for one face from its intersection records.
///
/// Walks the face edges in order, inserts intersection points in parametric
/// position along each edge, and emits the loop as a sequence of vertices.
///
/// @param subrange   Intersection records for this face
/// @param face       Range of vertex indices for this face
/// @param tag        Mesh tag for this face
/// @param get_point  (tag, id) -> point<int32_t, 3>; tag == -1 for intersection
/// @param get_flat_id (record) -> flat index in intersections buffer
/// @param work       Reusable scratch buffer
/// @param buffer     Output vertex buffer (appended to)
template <typename Index, typename Subrange, typename Face, typename GetPoint,
          typename GetFlatId>
auto extract_loop(const Subrange &subrange, const Face &face, Index tag,
                  const GetPoint &get_point, const GetFlatId &get_flat_id,
                  tf::buffer<loop_node<Index>> &work,
                  tf::buffer<vertex<Index>> &buffer) -> void {
  using i128 = tf::exact::int128;
  work.clear();
  for (const auto &rec : subrange) {
    if (rec.target.label == tf::topo_type::face)
      continue;
    work.push_back(
        {rec.target.id, rec.target.label, rec.id, get_flat_id(rec), i128(0)});
  }

  std::sort(work.begin(), work.end(), [](const auto &a, const auto &b) {
    return std::make_tuple(a.target_id, a.label, a.point_id) <
           std::make_tuple(b.target_id, b.label, b.point_id);
  });
  work.erase_till_end(
      std::unique(work.begin(), work.end(), [](const auto &a, const auto &b) {
        return a.target_id == b.target_id && a.label == b.label &&
               a.point_id == b.point_id;
      }));

  auto wit = work.begin();
  auto wend = work.end();
  Index n_edges = face.size();

  for (Index ei = 0; ei < n_edges; ++ei) {
    Index vi0 = face[ei];
    Index vi1 = face[tf::circular_increment(ei, n_edges)];

    auto edge_begin = wit;
    while (wit != wend && wit->target_id == ei)
      ++wit;

    if (edge_begin == wit) {
      buffer.push_back(
          {vertex_source::original, vi0, {short(ei), tf::topo_type::vertex}});
      continue;
    }

    auto p0 = get_point(tag, vi0);
    auto p1 = get_point(tag, vi1);
    i128 dx = i128(p1[0]) - p0[0];
    i128 dy = i128(p1[1]) - p0[1];
    i128 dz = i128(p1[2]) - p0[2];

    for (auto jt = edge_begin; jt != wit; ++jt) {
      auto p = get_point(-1, jt->point_id);
      jt->t = dx * (i128(p[0]) - p0[0]) + dy * (i128(p[1]) - p0[1]) +
              dz * (i128(p[2]) - p0[2]);
    }

    bool fwd = vi0 < vi1;
    std::sort(edge_begin, wit, [fwd](const auto &a, const auto &b) {
      if (a.label != b.label)
        return a.label < b.label;
      if (a.t != b.t)
        return a.t < b.t;
      return fwd ? a.point_id < b.point_id : a.point_id > b.point_id;
    });

    if (edge_begin->label != tf::topo_type::vertex)
      buffer.push_back(
          {vertex_source::original, vi0, {short(ei), tf::topo_type::vertex}});

    for (auto jt = edge_begin; jt != wit; ++jt)
      buffer.push_back({vertex_source::created,
                        jt->point_id,
                        {short(jt->target_id), jt->label}});
  }
}

} // namespace tf::intersect::graph
