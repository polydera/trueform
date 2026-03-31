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
#include "../../exact/orient3d.hpp"
#include "../../exact/segment_plane_intersect.hpp"
#include "../../exact/vertex.hpp"
#include "./emit_record.hpp"

namespace tf::exact {

/// Process edges with strict +/- plane signs against a face. For each
/// crossing, classifies via fan orient3d as EF, EE, or VE.
///
/// Fan triangle t = (A[0], A[t+1], A[t+2]):
///   v1 = orient3d(A[0],   A[t+1], D, E)  — real edge iff t==0
///   v2 = orient3d(A[t+1], A[t+2], D, E)  — always real edge (t+1)
///   v3 = orient3d(A[0],   A[t+2], D, E)  — real edge iff t==n-3
///
/// Classification is fully determined by exact orient3d values.
/// segment_plane_intersect is only used to compute the point coordinates.
template <typename Index, typename Int, typename EdgeIsRep, typename FaceIsRep,
          typename Intersections, typename Pts>
void crossing_edges_vs_face(
    const tf::buffer<tf::exact::vertex<Index, Int>> &edge_verts,
    std::size_t n_edge,
    const tf::buffer<tf::exact::vertex<Index, Int>> &face_verts,
    std::size_t n_face, const tf::small_vector<int, 16> &edge_signs,
    int edge_tag, int face_tag, Index edge_face_id, Index face_id,
    const EdgeIsRep &edge_is_rep, const FaceIsRep &face_is_rep,
    Intersections &intersections, Pts &pts, bool both_crossing) {
  for (std::size_t i = 0; i < n_edge; ++i) {
    if (!edge_is_rep(i).second)
      continue;
    auto ni = tf::circular_increment(i, n_edge);
    if (edge_signs[i] * edge_signs[ni] >= 0)
      continue;
    const auto &D = edge_verts[i], &E = edge_verts[ni];

    bool found = false;
    for (std::size_t t = 0; t + 2 < n_face && !found; ++t) {
      auto v1 =
          orient3d_value(face_verts[0].pt, face_verts[t + 1].pt, D.pt, E.pt);
      auto v2 = orient3d_value(face_verts[t + 1].pt, face_verts[t + 2].pt, D.pt,
                               E.pt);
      auto v3 =
          orient3d_value(face_verts[0].pt, face_verts[t + 2].pt, D.pt, E.pt);

      int n_zero = (v1 == 0) + (v2 == 0) + (v3 == 0);

      if (n_zero == 0) {
        bool s1 = v1 > 0, s2 = v2 > 0, s3 = v3 > 0;
        if (s1 == s2 && s2 != s3) {
          auto P =
              *segment_plane_intersect(face_verts[0].pt, face_verts[t + 1].pt,
                                       face_verts[t + 2].pt, D, E);
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {face_id, tf::topo_type::face}, P, intersections, pts);
          found = true;
        }
        continue;
      }

      if (n_zero >= 3)
        continue;

      if (n_zero == 2) {
        std::size_t k = (v1 == 0 && v2 == 0)   ? t + 1
                        : (v1 == 0 && v3 == 0) ? 0
                                               : /* v2==0 && v3==0 */ t + 2;
        if (face_is_rep(k).first)
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(k), tf::topo_type::vertex}, face_verts[k].pt,
                      intersections, pts);
        found = true;
        continue;
      }

      // n_zero == 1: check containment on the other two edges
      bool s1 = v1 > 0, s2 = v2 > 0, s3 = v3 > 0;
      bool contained = (v1 == 0)   ? (s2 != s3)
                       : (v2 == 0) ? (s1 != s3)
                                   : (s1 == s2); // v3==0
      if (!contained)
        continue;

      auto P = *segment_plane_intersect(face_verts[0].pt, face_verts[t + 1].pt,
                                        face_verts[t + 2].pt, D, E);

      bool v1_real = (t == 0);
      bool v3_real = (t + 3 == n_face);
      bool on_real_edge =
          (v1 == 0 && v1_real) || (v2 == 0) || (v3 == 0 && v3_real);

      if (on_real_edge) {
        std::size_t edge_k = (v2 == 0) ? t + 1 : (v1 == 0) ? 0 : n_face - 1;
        bool erep = edge_is_rep(i).second;
        bool ferep = face_is_rep(edge_k).second;
        if (erep && ferep && (!both_crossing || edge_tag < face_tag))
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(edge_k), tf::topo_type::edge}, P, intersections,
                      pts);
      } else {
        emit_record(edge_tag, face_tag, edge_face_id, face_id,
                    {Index(i), tf::topo_type::edge},
                    {face_id, tf::topo_type::face}, P, intersections, pts);
      }
      found = true;
    }
  }
}

/// Self-intersection variant: uses face ID ordering for EE dedup
/// instead of tag ordering (since both tags are equal).
template <typename Index, typename Int, typename EdgeIsRep, typename FaceIsRep,
          typename Intersections, typename Pts>
void crossing_edges_vs_face_self(
    const tf::buffer<tf::exact::vertex<Index, Int>> &edge_verts,
    std::size_t n_edge,
    const tf::buffer<tf::exact::vertex<Index, Int>> &face_verts,
    std::size_t n_face, const tf::small_vector<int, 16> &edge_signs,
    int edge_tag, int face_tag, Index edge_face_id, Index face_id,
    const EdgeIsRep &edge_is_rep, const FaceIsRep &face_is_rep,
    Intersections &intersections, Pts &pts, bool both_crossing) {
  for (std::size_t i = 0; i < n_edge; ++i) {
    if (!edge_is_rep(i).second)
      continue;
    auto ni = tf::circular_increment(i, n_edge);
    if (edge_signs[i] * edge_signs[ni] >= 0)
      continue;
    const auto &D = edge_verts[i], &E = edge_verts[ni];

    bool found = false;
    for (std::size_t t = 0; t + 2 < n_face && !found; ++t) {
      auto v1 =
          orient3d_value(face_verts[0].pt, face_verts[t + 1].pt, D.pt, E.pt);
      auto v2 = orient3d_value(face_verts[t + 1].pt, face_verts[t + 2].pt, D.pt,
                               E.pt);
      auto v3 =
          orient3d_value(face_verts[0].pt, face_verts[t + 2].pt, D.pt, E.pt);

      int n_zero = (v1 == 0) + (v2 == 0) + (v3 == 0);

      if (n_zero == 0) {
        bool s1 = v1 > 0, s2 = v2 > 0, s3 = v3 > 0;
        if (s1 == s2 && s2 != s3) {
          auto P =
              *segment_plane_intersect(face_verts[0].pt, face_verts[t + 1].pt,
                                       face_verts[t + 2].pt, D, E);
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {face_id, tf::topo_type::face}, P, intersections, pts);
          found = true;
        }
        continue;
      }

      if (n_zero >= 3)
        continue;

      if (n_zero == 2) {
        std::size_t k = (v1 == 0 && v2 == 0)   ? t + 1
                        : (v1 == 0 && v3 == 0) ? 0
                                               : /* v2==0 && v3==0 */ t + 2;
        if (face_is_rep(k).first)
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(k), tf::topo_type::vertex}, face_verts[k].pt,
                      intersections, pts);
        found = true;
        continue;
      }

      bool s1 = v1 > 0, s2 = v2 > 0, s3 = v3 > 0;
      bool contained = (v1 == 0)   ? (s2 != s3)
                       : (v2 == 0) ? (s1 != s3)
                                   : (s1 == s2);
      if (!contained)
        continue;

      auto P = *segment_plane_intersect(face_verts[0].pt, face_verts[t + 1].pt,
                                        face_verts[t + 2].pt, D, E);

      bool v1_real = (t == 0);
      bool v3_real = (t + 3 == n_face);
      bool on_real_edge =
          (v1 == 0 && v1_real) || (v2 == 0) || (v3 == 0 && v3_real);

      if (on_real_edge) {
        std::size_t edge_k = (v2 == 0) ? t + 1 : (v1 == 0) ? 0 : n_face - 1;
        bool erep = edge_is_rep(i).second;
        bool ferep = face_is_rep(edge_k).second;
        if (erep && ferep && (!both_crossing || edge_face_id < face_id))
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(edge_k), tf::topo_type::edge}, P, intersections,
                      pts);
      } else {
        emit_record(edge_tag, face_tag, edge_face_id, face_id,
                    {Index(i), tf::topo_type::edge},
                    {face_id, tf::topo_type::face}, P, intersections, pts);
      }
      found = true;
    }
  }
}

} // namespace tf::exact
