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
#include "../../core/small_vector.hpp"
#include "../../exact/coplanar_edge_edge_point.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/segments_cross.hpp"
#include "../../exact/vertex.hpp"
#include "./emit_record.hpp"
#include "./face_plane_info.hpp"

namespace tf::exact {

/// Detect intersections between features that lie on the other face's
/// plane (sign==0). Handles VV (shared coords), VE (vertex on edge
/// interior), and coplanar EE (crossing edges with all endpoints
/// on-plane). Skips VE when the edge has strict +/− signs — those are
/// already caught by crossing_edges_vs_face.
template <typename Index, typename IsRep0, typename IsRep1, typename Ints,
          typename Pts>
void coplanar_primitives(
    const tf::buffer<vertex> &face0, std::size_t n0,
    const tf::buffer<vertex> &face1, std::size_t n1,
    const tf::small_vector<int, 16> &signs0,
    const tf::small_vector<int, 16> &signs1, int tag0, int tag1,
    Index face0_id, Index face1_id, const IsRep0 &is_rep0,
    const IsRep1 &is_rep1, const face_plane_info &plane0,
    const face_plane_info &plane1, Ints &ints, Pts &pts) {
  auto between = [](int32_t a, int32_t b, int32_t v) {
    return (a <= v && v <= b) || (b <= v && v <= a);
  };
  auto pts_equal = [](const pt3 &a, const pt3 &b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
  };
  for (std::size_t i = 0; i < n0; ++i) {
    auto ni = tf::circular_increment(i, n0);
    auto [vrep_i, erep_i] = is_rep0(i);
    for (std::size_t j = 0; j < n1; ++j) {
      auto nj = tf::circular_increment(j, n1);
      auto [vrep_j, erep_j] = is_rep1(j);

      // VV: vertex i == vertex j
      if (vrep_i && vrep_j && signs0[i] == 0 && signs1[j] == 0 &&
          face0[i].pt[0] == face1[j].pt[0] &&
          face0[i].pt[1] == face1[j].pt[1] &&
          face0[i].pt[2] == face1[j].pt[2]) {
        emit_record(
            tag0, tag1, face0_id, face1_id, {Index(i), tf::topo_type::vertex},
            {Index(j), tf::topo_type::vertex}, face0[i].pt, ints, pts);
      }

      // VE: vertex i of poly0 on edge j of poly1 (interior only, not
      // endpoints). Guard: only when edge j is NOT strict +/- (at least one
      // endpoint on poly0's plane). If strict +/-, crossing_edges_vs_face
      // handles it.
      if (vrep_i && erep_j && signs0[i] == 0 && signs1[j] * signs1[nj] == 0 &&
          plane1.valid && !pts_equal(face0[i].pt, face1[j].pt) &&
          !pts_equal(face0[i].pt, face1[nj].pt)) {
        if (orient2d_sign(face1[j], face1[nj], face0[i], plane1.ax0,
                          plane1.ax1) == 0 &&
            between(face1[j].pt[plane1.ax0], face1[nj].pt[plane1.ax0],
                    face0[i].pt[plane1.ax0]) &&
            between(face1[j].pt[plane1.ax1], face1[nj].pt[plane1.ax1],
                    face0[i].pt[plane1.ax1])) {
          emit_record(tag0, tag1, face0_id, face1_id,
                      {Index(i), tf::topo_type::vertex},
                      {Index(j), tf::topo_type::edge}, face0[i].pt, ints, pts);
        }
      }

      // VE: vertex j of poly1 on edge i of poly0 (interior only, not
      // endpoints). Guard: only when edge i is NOT strict +/- (at least one
      // endpoint on poly1's plane). If strict +/-, crossing_edges_vs_face
      // handles it.
      if (vrep_j && erep_i && signs1[j] == 0 && signs0[i] * signs0[ni] == 0 &&
          plane0.valid && !pts_equal(face1[j].pt, face0[i].pt) &&
          !pts_equal(face1[j].pt, face0[ni].pt)) {
        if (orient2d_sign(face0[i], face0[ni], face1[j], plane0.ax0,
                          plane0.ax1) == 0 &&
            between(face0[i].pt[plane0.ax0], face0[ni].pt[plane0.ax0],
                    face1[j].pt[plane0.ax0]) &&
            between(face0[i].pt[plane0.ax1], face0[ni].pt[plane0.ax1],
                    face1[j].pt[plane0.ax1])) {
          emit_record(tag1, tag0, face1_id, face0_id,
                      {Index(j), tf::topo_type::vertex},
                      {Index(i), tf::topo_type::edge}, face1[j].pt, ints, pts);
        }
      }

      // EE: coplanar edges cross
      if (erep_i && erep_j && signs0[i] == 0 && signs0[ni] == 0 &&
          signs1[j] == 0 && signs1[nj] == 0) {
        int ax0 = plane1.valid ? plane1.ax0 : plane0.ax0;
        int ax1 = plane1.valid ? plane1.ax1 : plane0.ax1;
        pt2 p0i = {face0[i].pt[ax0], face0[i].pt[ax1]};
        pt2 p0ni = {face0[ni].pt[ax0], face0[ni].pt[ax1]};
        pt2 p1j = {face1[j].pt[ax0], face1[j].pt[ax1]};
        pt2 p1nj = {face1[nj].pt[ax0], face1[nj].pt[ax1]};
        if (segments_cross(p0i, p0ni, p1j, p1nj)) {
          auto pt = coplanar_edge_edge_point(face0[i], face0[ni], face1[j],
                                             face1[nj], ax0, ax1);
          emit_record(tag0, tag1, face0_id, face1_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(j), tf::topo_type::edge}, pt, ints, pts);
        }
      }
    }
  }
}

} // namespace tf::exact
