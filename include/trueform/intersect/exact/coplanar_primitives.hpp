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
#include "../../core/small_vector.hpp"
#include "../../exact/coplanar_edge_edge_point.hpp"
#include "../../exact/segments_cross.hpp"
#include "../../exact/vertex.hpp"
#include "./emit_record.hpp"
#include "./face_plane_info.hpp"
#include "./predicate_kernel.hpp"

namespace tf::exact {

/// Detect intersections between features that lie on the other face's
/// plane (sign==0). Handles VV (shared coords), VE (vertex on edge
/// interior), and coplanar EE (crossing edges with all endpoints
/// on-plane). VE with a crossing edge is skipped — `crossing_edges_vs_face`
/// covers that case.
template <typename Index, typename Int, typename IsRep0, typename IsRep1,
          typename IsShared0, typename IsShared1, typename Intersections,
          typename Pts,
          typename Kernel = tf::exact::predicate_kernel<Int>>
void coplanar_primitives(
    tf::exact::vertex_range<Index, Int> face0, std::size_t n0,
    tf::exact::vertex_range<Index, Int> face1, std::size_t n1,
    const tf::small_vector<int, 16> &signs0,
    const tf::small_vector<int, 16> &signs1, int tag0, int tag1, Index face0_id,
    Index face1_id, const IsRep0 &is_rep0, const IsRep1 &is_rep1,
    const face_plane_info &plane0, const face_plane_info &plane1,
    const IsShared0 &is_shared0, const IsShared1 &is_shared1,
    Intersections &intersections, Pts &pts, const Kernel &kernel = {}) {
  auto between = [](Int a, Int b, Int v) {
    return (a <= v && v <= b) || (b <= v && v <= a);
  };

  for (std::size_t i = 0; i < n0; ++i) {
    auto ni = tf::circular_increment(i, n0);
    auto [vrep_i, erep_i] = is_rep0(i);
    auto sh_i = is_shared0(i);
    auto sh_ni = is_shared0(ni);
    for (std::size_t j = 0; j < n1; ++j) {
      auto nj = tf::circular_increment(j, n1);
      auto [vrep_j, erep_j] = is_rep1(j);
      auto sh_j = is_shared1(j);
      auto sh_nj = is_shared1(nj);

      // VV: vertex i == vertex j (skip topologically shared vertices)
      if (!(sh_i && sh_j) && vrep_i && vrep_j && signs0[i] == 0 &&
          signs1[j] == 0 && kernel.vv_equal(face0[i].pt, face1[j].pt)) {
        emit_record(
            tag0, tag1, face0_id, face1_id, {Index(i), tf::topo_type::vertex},
            {Index(j), tf::topo_type::vertex}, face0[i].pt, intersections, pts);
      }

      // VE: vertex i of poly0 on edge j of poly1
      if (!sh_i && vrep_i && erep_j && signs0[i] == 0 &&
          signs1[j] * signs1[nj] == 0 && plane1.valid &&
          !kernel.vv_equal(face0[i].pt, face1[j].pt) &&
          !kernel.vv_equal(face0[i].pt, face1[nj].pt)) {
        if (kernel.orient2d_sign(face1[j], face1[nj], face0[i], plane1.ax0,
                                 plane1.ax1) == 0 &&
            between(face1[j].pt[plane1.ax0], face1[nj].pt[plane1.ax0],
                    face0[i].pt[plane1.ax0]) &&
            between(face1[j].pt[plane1.ax1], face1[nj].pt[plane1.ax1],
                    face0[i].pt[plane1.ax1])) {
          emit_record(
              tag0, tag1, face0_id, face1_id, {Index(i), tf::topo_type::vertex},
              {Index(j), tf::topo_type::edge}, face0[i].pt, intersections, pts);
        }
      }

      // VE: vertex j of poly1 on edge i of poly0
      if (!sh_j && vrep_j && erep_i && signs1[j] == 0 &&
          signs0[i] * signs0[ni] == 0 && plane0.valid &&
          !kernel.vv_equal(face1[j].pt, face0[i].pt) &&
          !kernel.vv_equal(face1[j].pt, face0[ni].pt)) {
        if (kernel.orient2d_sign(face0[i], face0[ni], face1[j], plane0.ax0,
                                 plane0.ax1) == 0 &&
            between(face0[i].pt[plane0.ax0], face0[ni].pt[plane0.ax0],
                    face1[j].pt[plane0.ax0]) &&
            between(face0[i].pt[plane0.ax1], face0[ni].pt[plane0.ax1],
                    face1[j].pt[plane0.ax1])) {
          emit_record(
              tag1, tag0, face1_id, face0_id, {Index(j), tf::topo_type::vertex},
              {Index(i), tf::topo_type::edge}, face1[j].pt, intersections, pts);
        }
      }

      // EE: coplanar edges cross (skip if either edge IS the shared edge)
      if (!((sh_i && sh_ni) || (sh_j && sh_nj)) && erep_i && erep_j &&
          signs0[i] == 0 && signs0[ni] == 0 && signs1[j] == 0 &&
          signs1[nj] == 0) {
        int ax0 = plane1.valid ? plane1.ax0 : plane0.ax0;
        int ax1 = plane1.valid ? plane1.ax1 : plane0.ax1;
        pt2<Int> p0i = {face0[i].pt[ax0], face0[i].pt[ax1]};
        pt2<Int> p0ni = {face0[ni].pt[ax0], face0[ni].pt[ax1]};
        pt2<Int> p1j = {face1[j].pt[ax0], face1[j].pt[ax1]};
        pt2<Int> p1nj = {face1[nj].pt[ax0], face1[nj].pt[ax1]};
        if (tf::exact::segments_cross(p0i, p0ni, p1j, p1nj, kernel)) {
          auto pt = kernel.edge_edge_point(face0[i], face0[ni], face1[j],
                                           face1[nj], ax0, ax1);
          emit_record(tag0, tag1, face0_id, face1_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(j), tf::topo_type::edge}, pt, intersections, pts);
        }
      }
    }
  }
}

template <typename Index, typename Int, typename IsRep0, typename IsRep1,
          typename Intersections, typename Pts,
          typename Kernel = tf::exact::predicate_kernel<Int>>
void coplanar_primitives(
    tf::exact::vertex_range<Index, Int> face0, std::size_t n0,
    tf::exact::vertex_range<Index, Int> face1, std::size_t n1,
    const tf::small_vector<int, 16> &signs0,
    const tf::small_vector<int, 16> &signs1, int tag0, int tag1, Index face0_id,
    Index face1_id, const IsRep0 &is_rep0, const IsRep1 &is_rep1,
    const face_plane_info &plane0, const face_plane_info &plane1,
    Intersections &intersections, Pts &pts, const Kernel &kernel = {}) {
  auto no_shared = [](std::size_t) { return false; };
  coplanar_primitives(face0, n0, face1, n1, signs0, signs1, tag0, tag1,
                      face0_id, face1_id, is_rep0, is_rep1, plane0, plane1,
                      no_shared, no_shared, intersections, pts, kernel);
}

} // namespace tf::exact
