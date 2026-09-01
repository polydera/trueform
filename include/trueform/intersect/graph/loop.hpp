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
#include "../../exact/meta.hpp"
#include "../../topology/topo_type.hpp"
#include "./vertex.hpp"

#include <algorithm>
#include <tuple>

namespace tf::intersect::graph {

/// Work node for parametric sorting of intersection points along an edge.
template <typename Index, typename Int> struct loop_node {
  Index target_id;
  tf::topo_type label;
  Index point_id;
  typename tf::exact::meta<Int>::T2 t;
};

/// Extract a boundary loop for one face from what the fan told it.
///
/// Walks the face edges in order, inserts intersection points in parametric
/// position along each edge, and emits the loop as a sequence of vertices.
///
/// The face's claims arrive on both of the fan's currencies — a pair
/// record names the point at this face as well as the pair it stands on,
/// a delivery names only the point — so the loop reads BOTH and keeps one
/// representative per distinct (target, point), which is the same
/// contract either currency alone was read under.
///
/// @param records    Pair records for this face
/// @param deliveries Point deliveries for this face
/// @param face       Range of vertex indices for this face
/// @param tag        Mesh tag for this face
/// @param get_point  (tag, id) -> point<Int, 3>; tag == -1 for intersection
/// @param work       Reusable scratch buffer
/// @param buffer     Output vertex buffer (appended to)
template <typename Index, typename Int, typename Records, typename Deliveries,
          typename Face, typename GetPoint>
auto extract_loop(const Records &records, const Deliveries &deliveries,
                  const Face &face, Index tag, const GetPoint &get_point,
                  tf::buffer<loop_node<Index, Int>> &work,
                  tf::buffer<vertex<Index>> &buffer) -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  work.clear();
  const auto claim = [&](const auto &rec) {
    if (rec.target.label == tf::topo_type::face)
      return;
    work.push_back({rec.target.id, rec.target.label, rec.id, T2(0)});
  };
  for (const auto &rec : records)
    claim(rec);
  for (const auto &rec : deliveries)
    claim(rec);

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
    T1 dx = T1(p1[0]) - p0[0];
    T1 dy = T1(p1[1]) - p0[1];
    T1 dz = T1(p1[2]) - p0[2];

    for (auto jt = edge_begin; jt != wit; ++jt) {
      auto p = get_point(-1, jt->point_id);
      jt->t = T2(dx) * (T1(p[0]) - p0[0]) + T2(dy) * (T1(p[1]) - p0[1]) +
              T2(dz) * (T1(p[2]) - p0[2]);
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
