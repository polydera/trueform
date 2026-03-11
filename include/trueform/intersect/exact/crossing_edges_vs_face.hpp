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
#include "./classify_crossing.hpp"
#include "./emit_record.hpp"
#include "./face_plane_info.hpp"

namespace tf::exact {

/// Process edges with strict +/− plane signs against a face. For each
/// crossing, computes the plane intersection point and classifies it as
/// EF, EE, or VE via classify_crossing.
template <typename Index, typename IsRep, typename MEL, typename Ints,
          typename Pts>
void crossing_edges_vs_face(
    const tf::buffer<vertex> &edge_verts, std::size_t n_edge,
    const tf::buffer<vertex> &face_verts, std::size_t n_face,
    const tf::small_vector<int, 16> &edge_signs, int edge_tag, int face_tag,
    Index edge_face_id, Index face_id, const MEL &mel,
    const IsRep &face_is_rep, const face_plane_info &fplane, Ints &ints,
    Pts &pts) {
  for (std::size_t i = 0; i < n_edge; ++i) {
    if (!mel[edge_face_id][i].is_representative(edge_face_id))
      continue;
    auto ni = tf::circular_increment(i, n_edge);
    if (edge_signs[i] * edge_signs[ni] >= 0)
      continue;
    const auto &D = edge_verts[i], &E = edge_verts[ni];

    // Fan triangulation: if all orient3d_values are nonzero, the
    // crossing is strictly inside a fan triangle → EF. Any zero means
    // the crossing is on a real or fictitious fan edge → fall through
    // to classify_crossing for exact 2D classification (EF/EE/VE).
    bool fast_found = false;
    for (std::size_t t = 0; t + 2 < n_face; ++t) {
      auto v1 =
          orient3d_value(face_verts[0].pt, face_verts[t + 1].pt, D.pt, E.pt);
      auto v2 = orient3d_value(face_verts[t + 1].pt, face_verts[t + 2].pt,
                               D.pt, E.pt);
      auto v3 =
          orient3d_value(face_verts[0].pt, face_verts[t + 2].pt, D.pt, E.pt);
      if (v1 == 0 || v2 == 0 || v3 == 0)
        break;
      bool s1 = v1 > 0, s2 = v2 > 0, s3 = v3 > 0;
      if (s1 == s2 && s2 != s3) {
        auto pt =
            *segment_plane_intersect(face_verts[0].pt, face_verts[t + 1].pt,
                                     face_verts[t + 2].pt, D, E);
        emit_record(edge_tag, face_tag, edge_face_id, face_id,
                    {Index(i), tf::topo_type::edge},
                    {face_id, tf::topo_type::face}, pt, ints, pts);
        fast_found = true;
        break;
      }
    }
    if (fast_found)
      continue;

    // Fan had a zero or no containment → classify in 2D on real edges
    if (!fplane.valid)
      continue;
    auto pt_opt = segment_plane_intersect(face_verts[fplane.i0].pt,
                                          face_verts[fplane.i1].pt,
                                          face_verts[fplane.i2].pt, D, E);
    if (!pt_opt)
      continue;
    auto P = *pt_opt;
    classify_crossing(P, face_verts, n_face, fplane.ax0, fplane.ax1, edge_tag,
                      face_tag, edge_face_id, face_id, Index(i), face_is_rep,
                      ints, pts);
  }
}

} // namespace tf::exact
