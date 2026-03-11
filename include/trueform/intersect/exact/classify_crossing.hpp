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
#include "../../exact/orient2d.hpp"
#include "../../exact/vertex.hpp"
#include "./emit_record.hpp"

namespace tf::exact {

/// Classify where a plane-crossing point P lands on a face: strictly inside
/// (EF), on an edge interior (EE), or at a vertex (VE). Projects to 2D
/// using the face's dominant plane.
template <typename Index, typename IsRep, typename Ints, typename Pts>
void classify_crossing(const pt3 &P, const tf::buffer<vertex> &face_verts,
                       std::size_t n_face, int ax0, int ax1, int edge_tag,
                       int face_tag, Index edge_face_id, Index face_id,
                       Index edge_local_idx, const IsRep &face_is_rep,
                       Ints &ints, Pts &pts) {
  // Check if P is at a face vertex (edge-through-vertex)
  for (std::size_t k = 0; k < n_face; ++k) {
    if (P[0] == face_verts[k].pt[0] && P[1] == face_verts[k].pt[1] &&
        P[2] == face_verts[k].pt[2]) {
      if (face_is_rep(k).first) {
        emit_record(edge_tag, face_tag, edge_face_id, face_id,
                    {edge_local_idx, tf::topo_type::edge},
                    {Index(k), tf::topo_type::vertex}, face_verts[k].pt, ints,
                    pts);
      }
      return;
    }
  }
  // Check if P is on a face edge (EE — symmetric, only from edge_tag <
  // face_tag). When edge_tag >= face_tag, skip entirely: the other direction
  // handles it. The EF check below correctly rejects on-edge points (orient2d
  // == 0 → inside = false).
  vertex Pv{0, P};
  if (edge_tag < face_tag) {
    for (std::size_t k = 0; k < n_face; ++k) {
      auto nk = tf::circular_increment(k, n_face);
      if (orient2d_sign(face_verts[k], face_verts[nk], Pv, ax0, ax1) == 0) {
        auto between = [](int32_t a, int32_t b, int32_t v) {
          return (a <= v && v <= b) || (b <= v && v <= a);
        };
        if (between(face_verts[k].pt[ax0], face_verts[nk].pt[ax0], P[ax0]) &&
            between(face_verts[k].pt[ax1], face_verts[nk].pt[ax1], P[ax1])) {
          if (face_is_rep(k).second)
            emit_record(edge_tag, face_tag, edge_face_id, face_id,
                        {edge_local_idx, tf::topo_type::edge},
                        {Index(k), tf::topo_type::edge}, P, ints, pts);
          return;
        }
      }
    }
  }
  // P is strictly inside the face (EF)
  int first_sign = 0;
  bool inside = true;
  for (std::size_t k = 0; k < n_face && inside; ++k) {
    auto nk = tf::circular_increment(k, n_face);
    auto s = orient2d_sign(face_verts[k], face_verts[nk], Pv, ax0, ax1);
    if (s == 0)
      inside = false;
    else if (first_sign == 0)
      first_sign = s;
    else if (s != first_sign)
      inside = false;
  }
  if (inside && first_sign != 0)
    emit_record(edge_tag, face_tag, edge_face_id, face_id,
                {edge_local_idx, tf::topo_type::edge},
                {face_id, tf::topo_type::face}, P, ints, pts);
}

} // namespace tf::exact
