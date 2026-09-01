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
#include "../../exact/meta.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/segment_plane_intersect.hpp"
#include "../../exact/vertex.hpp"
#include "./emit_record.hpp"
#include "./intersection_payload.hpp"

namespace tf::exact {

/// Process edges with strict +/- plane signs against a face. For each
/// crossing, classifies via fan orient3d as EF, EE, or VE.
///
/// Fan triangle t = (A[0], A[t+1], A[t+2]):
///   v1 = orient3d(A[0],   A[t+1], D, E)  — real edge iff t==0
///   v2 = orient3d(A[t+1], A[t+2], D, E)  — always real edge (t+1)
///   v3 = orient3d(A[0],   A[t+2], D, E)  — real edge iff t==n-3
///
/// `edge_values` are the face plane's orient3d values behind
/// `edge_signs`; only a parameter payload reads them, and then they are
/// exactly the pierce's fraction along its edge.
template <typename Index, typename Int, typename EdgeIsRep, typename FaceIsRep,
          typename Intersections, typename Pts>
void crossing_edges_vs_face(
    tf::exact::vertex_range<Index, Int> edge_verts, std::size_t n_edge,
    tf::exact::vertex_range<Index, Int> face_verts, std::size_t n_face,
    const tf::small_vector<int, 16> &edge_signs,
    const tf::small_vector<typename tf::exact::meta<Int>::T2, 16> &edge_values,
    int edge_tag, int face_tag, Index edge_face_id, Index face_id,
    const EdgeIsRep &edge_is_rep, const FaceIsRep &face_is_rep,
    Intersections &intersections, Pts &pts, bool both_crossing) {
  using payload_t = typename Pts::value_type;
  for (std::size_t i = 0; i < n_edge; ++i) {
    if (!edge_is_rep(i).second)
      continue;
    auto ni = tf::circular_increment(i, n_edge);
    if (edge_signs[i] * edge_signs[ni] >= 0)
      continue;
    const auto &D = edge_verts[i], &E = edge_verts[ni];

    bool found = false;
    for (std::size_t t = 0; t + 2 < n_face && !found; ++t) {
      int s1 =
          orient3d_sign(face_verts[0].pt, face_verts[t + 1].pt, D.pt, E.pt);
      int s2 =
          orient3d_sign(face_verts[t + 1].pt, face_verts[t + 2].pt, D.pt, E.pt);
      int s3 =
          orient3d_sign(face_verts[0].pt, face_verts[t + 2].pt, D.pt, E.pt);

      int n_zero = (s1 == 0) + (s2 == 0) + (s3 == 0);

      if (n_zero == 0) {
        bool p1 = s1 > 0, p2 = s2 > 0, p3 = s3 > 0;
        if (p1 == p2 && p2 != p3) {
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {face_id, tf::topo_type::face},
                      make_edge_plane_payload<payload_t>(
                          face_verts[0].pt, face_verts[t + 1].pt,
                          face_verts[t + 2].pt, D, E, edge_values, i, ni),
                      intersections, pts);
          found = true;
        }
        continue;
      }

      if (n_zero >= 3)
        continue;

      if (n_zero == 2) {
        std::size_t k = (s1 == 0 && s2 == 0)   ? t + 1
                        : (s1 == 0 && s3 == 0) ? 0
                                               : /* s2==0 && s3==0 */ t + 2;
        if (face_is_rep(k).first)
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(k), tf::topo_type::vertex},
                      make_vertex_edge_payload<payload_t>(
                          D, E, face_verts[k].pt, face_verts[k].id),
                      intersections, pts);
        found = true;
        continue;
      }

      // n_zero == 1: check containment on the other two edges
      bool p1 = s1 > 0, p2 = s2 > 0, p3 = s3 > 0;
      bool contained = (s1 == 0)   ? (p2 != p3)
                       : (s2 == 0) ? (p1 != p3)
                                   : (p1 == p2); // s3==0
      if (!contained)
        continue;

      bool v1_real = (t == 0);
      bool v3_real = (t + 3 == n_face);
      bool on_real_edge =
          (s1 == 0 && v1_real) || (s2 == 0) || (s3 == 0 && v3_real);

      std::size_t edge_k = 0;
      std::size_t nk = 0;
      if (on_real_edge) {
        edge_k = (s2 == 0) ? t + 1 : (s1 == 0) ? 0 : n_face - 1;
        nk = tf::circular_increment(edge_k, n_face);
      }
      if (on_real_edge) {
        bool erep = edge_is_rep(i).second;
        bool ferep = face_is_rep(edge_k).second;
        if (erep && ferep && (!both_crossing || edge_tag < face_tag))
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(edge_k), tf::topo_type::edge},
                      make_edge_edge_payload<payload_t>(
                          D, E, face_verts[edge_k], face_verts[nk],
                          [&] {
                            return *segment_plane_intersect(
                                face_verts[0].pt, face_verts[t + 1].pt,
                                face_verts[t + 2].pt, D, E);
                          }),
                      intersections, pts);
      } else {
        emit_record(edge_tag, face_tag, edge_face_id, face_id,
                    {Index(i), tf::topo_type::edge},
                    {face_id, tf::topo_type::face},
                    make_edge_plane_payload<payload_t>(
                        face_verts[0].pt, face_verts[t + 1].pt,
                        face_verts[t + 2].pt, D, E, edge_values, i, ni),
                    intersections, pts);
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
    tf::exact::vertex_range<Index, Int> edge_verts, std::size_t n_edge,
    tf::exact::vertex_range<Index, Int> face_verts, std::size_t n_face,
    const tf::small_vector<int, 16> &edge_signs,
    const tf::small_vector<typename tf::exact::meta<Int>::T2, 16> &edge_values,
    int edge_tag, int face_tag, Index edge_face_id, Index face_id,
    const EdgeIsRep &edge_is_rep, const FaceIsRep &face_is_rep,
    Intersections &intersections, Pts &pts, bool both_crossing) {
  using payload_t = typename Pts::value_type;
  for (std::size_t i = 0; i < n_edge; ++i) {
    if (!edge_is_rep(i).second)
      continue;
    auto ni = tf::circular_increment(i, n_edge);
    if (edge_signs[i] * edge_signs[ni] >= 0)
      continue;
    const auto &D = edge_verts[i], &E = edge_verts[ni];

    bool found = false;
    for (std::size_t t = 0; t + 2 < n_face && !found; ++t) {
      int s1 =
          orient3d_sign(face_verts[0].pt, face_verts[t + 1].pt, D.pt, E.pt);
      int s2 =
          orient3d_sign(face_verts[t + 1].pt, face_verts[t + 2].pt, D.pt, E.pt);
      int s3 =
          orient3d_sign(face_verts[0].pt, face_verts[t + 2].pt, D.pt, E.pt);

      int n_zero = (s1 == 0) + (s2 == 0) + (s3 == 0);

      if (n_zero == 0) {
        bool p1 = s1 > 0, p2 = s2 > 0, p3 = s3 > 0;
        if (p1 == p2 && p2 != p3) {
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {face_id, tf::topo_type::face},
                      make_edge_plane_payload<payload_t>(
                          face_verts[0].pt, face_verts[t + 1].pt,
                          face_verts[t + 2].pt, D, E, edge_values, i, ni),
                      intersections, pts);
          found = true;
        }
        continue;
      }

      if (n_zero >= 3)
        continue;

      if (n_zero == 2) {
        std::size_t k = (s1 == 0 && s2 == 0)   ? t + 1
                        : (s1 == 0 && s3 == 0) ? 0
                                               : /* s2==0 && s3==0 */ t + 2;
        if (face_is_rep(k).first)
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(k), tf::topo_type::vertex},
                      make_vertex_edge_payload<payload_t>(
                          D, E, face_verts[k].pt, face_verts[k].id),
                      intersections, pts);
        found = true;
        continue;
      }

      bool p1 = s1 > 0, p2 = s2 > 0, p3 = s3 > 0;
      bool contained = (s1 == 0)   ? (p2 != p3)
                       : (s2 == 0) ? (p1 != p3)
                                   : (p1 == p2);
      if (!contained)
        continue;

      bool v1_real = (t == 0);
      bool v3_real = (t + 3 == n_face);
      bool on_real_edge =
          (s1 == 0 && v1_real) || (s2 == 0) || (s3 == 0 && v3_real);

      std::size_t edge_k = 0;
      std::size_t nk = 0;
      if (on_real_edge) {
        edge_k = (s2 == 0) ? t + 1 : (s1 == 0) ? 0 : n_face - 1;
        nk = tf::circular_increment(edge_k, n_face);
      }
      if (on_real_edge) {
        bool erep = edge_is_rep(i).second;
        bool ferep = face_is_rep(edge_k).second;
        if (erep && ferep && (!both_crossing || edge_face_id < face_id))
          emit_record(edge_tag, face_tag, edge_face_id, face_id,
                      {Index(i), tf::topo_type::edge},
                      {Index(edge_k), tf::topo_type::edge},
                      make_edge_edge_payload<payload_t>(
                          D, E, face_verts[edge_k], face_verts[nk],
                          [&] {
                            return *segment_plane_intersect(
                                face_verts[0].pt, face_verts[t + 1].pt,
                                face_verts[t + 2].pt, D, E);
                          }),
                      intersections, pts);
      } else {
        emit_record(edge_tag, face_tag, edge_face_id, face_id,
                    {Index(i), tf::topo_type::edge},
                    {face_id, tf::topo_type::face},
                    make_edge_plane_payload<payload_t>(
                        face_verts[0].pt, face_verts[t + 1].pt,
                        face_verts[t + 2].pt, D, E, edge_values, i, ni),
                    intersections, pts);
      }
      found = true;
    }
  }
}

} // namespace tf::exact
