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
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/intersects.hpp"
#include "../../core/small_vector.hpp"
#include "../../exact/triangle_segment_intersection.hpp"
#include "../../exact/vertex.hpp"
#include "../exact/coplanar_primitives.hpp"
#include "../exact/crossing_edges_vs_face.hpp"
#include "../exact/face_plane_info.hpp"
#include "../exact/predicate_kernel.hpp"
#include "../exact/vertex_face.hpp"
#include "./face_pair_search.hpp"

#include <array>
#include <optional>
#include <utility>

namespace tf::intersect {

/// First-hit SoS intersection of a segment against a convex face's
/// triangle fan. Shared by the pair and self SoS kernels.
template <typename Index, typename Int>
auto edge_vs_convex_face_sos(tf::exact::vertex_range<Index, Int> face,
                             const tf::exact::vertex<Index, Int> &v0,
                             const tf::exact::vertex<Index, Int> &v1)
    -> std::optional<tf::exact::pt3<Int>> {
  auto n = face.size();
  for (std::size_t t = 0; t + 2 < n; ++t) {
    if (auto pt = tf::exact::triangle_segment_intersect_point_sos(
            std::array<tf::exact::vertex<Index, Int>, 5>{
                face[0], face[t + 1], face[t + 2], v0, v1}))
      return pt;
  }
  return std::nullopt;
}

/// Cross-pair SoS: representative edges of one face vs the other face.
template <typename Index, typename Int, typename EdgeIsRep, typename Ints,
          typename Pts>
auto edges_vs_face_sos(tf::exact::vertex_range<Index, Int> edge_verts,
                       Index edge_id,
                       tf::exact::vertex_range<Index, Int> face_verts,
                       Index face_id, int edge_tag, int face_tag,
                       const EdgeIsRep &edge_is_rep, Ints &ints, Pts &pts) {
  auto n = edge_verts.size();
  for (std::size_t j = 0; j < n; ++j) {
    if (!edge_is_rep(j))
      continue;
    auto next_j = tf::circular_increment(j, n);
    if (auto pt = edge_vs_convex_face_sos(face_verts, edge_verts[j],
                                          edge_verts[next_j])) {
      Index id = pts.size();
      pts.push_back(*pt);
      auto edge_target = tf::topo_id<Index>{Index(j), tf::topo_type::edge};
      auto face_target = tf::topo_id<Index>{face_id, tf::topo_type::face};
      ints.push_back({short(edge_tag), short(face_tag), edge_id, face_id,
                      edge_target, face_target, id});
    }
  }
}

/// Cross-pair SoS over a prepped workspace leaf pair.
template <typename Index, typename Int, typename MEL0, typename MEL1>
void sos_process(face_pair_workspace<Index, Int> &ws, int tag0, int tag1,
                 const MEL0 &mel0, const MEL1 &mel1) {
  auto n0 = ws.n0();
  auto n1 = ws.n1();
  for (std::size_t i = 0; i < n0; ++i)
    for (std::size_t j = 0; j < n1; ++j) {
      if (!tf::intersects(ws.ibox0[i], ws.ibox1[j]))
        continue;
      auto id0 = Index(ws.ids0[i]);
      auto id1 = Index(ws.ids1[j]);
      auto erep0 = [&](std::size_t e) {
        return mel0[id0][e].is_representative(id0);
      };
      auto erep1 = [&](std::size_t e) {
        return mel1[id1][e].is_representative(id1);
      };
      edges_vs_face_sos(ws.face0(i), id0, ws.face1(j), id1, tag0, tag1, erep0,
                        ws.intersections, ws.points);
      edges_vs_face_sos(ws.face1(j), id1, ws.face0(i), id0, tag1, tag0, erep1,
                        ws.intersections, ws.points);
    }
}

/// Cross-pair primitives classification of two prepped faces (ranges +
/// cached planes). EF / crossing-EE / crossing-VE, then the coplanar
/// family and VF when a zero sign appears.
template <typename Index, typename Int, typename Poly0, typename Poly1,
          typename MEL0, typename MEL1, typename FM0, typename FM1,
          typename Ints, typename Pts>
auto primitives_polygon_pair_prepped(
    const Poly0 &poly0, const Poly1 &poly1, int tag0, int tag1,
    const MEL0 &mel0, const MEL1 &mel1, const FM0 &fm0, const FM1 &fm1,
    tf::exact::vertex_range<Index, Int> face_buf0,
    const tf::exact::face_plane<Int> &fp0,
    tf::exact::vertex_range<Index, Int> face_buf1,
    const tf::exact::face_plane<Int> &fp1, Ints &ints, Pts &pts,
    const tf::exact::predicate_kernel<Int> &kernel = {}) {
  auto face0_id = Index(poly0.id());
  auto face1_id = Index(poly1.id());

  auto n0 = face_buf0.size();
  auto n1 = face_buf1.size();

  const auto &plane0 = fp0.info;
  const auto &plane1 = fp1.info;

  // Compute orient3d_sign for all vertices vs opposite face plane.
  // sign_mask bits: 0 = has_negative, 1 = has_zero, 2 = has_positive.
  constexpr int has_negative = 1 << 0;
  constexpr int has_zero = 1 << 1;
  constexpr int has_positive = 1 << 2;
  constexpr int has_crossing = has_negative | has_positive;

  tf::small_vector<int, 16> signs0, signs1;
  signs0.resize(n0);
  signs1.resize(n1);
  int mask0 = 0, mask1 = 0;
  if (plane1.valid) {
    for (decltype(n0) i = 0; i < n0; ++i) {
      signs0[i] = tf::exact::orient3d_plane_sign(fp1.plane, face_buf0[i].pt);
      mask0 |= 1 << (signs0[i] + 1);
    }
  }
  if (plane0.valid) {
    for (decltype(n1) j = 0; j < n1; ++j) {
      signs1[j] = tf::exact::orient3d_plane_sign(fp0.plane, face_buf1[j].pt);
      mask1 |= 1 << (signs1[j] + 1);
    }
  }

  // Separating-plane reject. The both-sided form (no zero, neither face
  // crosses the other's plane) holds under a tolerance band. The
  // either-one-sided form prunes more but is sound only at tolerance 0: under
  // a band a strictly-separated-but-within-band pair must still snap.
  int combined = mask0 | mask1;
  bool both_separated = !(combined & has_zero) &&
                        (mask0 & has_crossing) != has_crossing &&
                        (mask1 & has_crossing) != has_crossing;
  bool either_strict = (mask0 == has_positive) || (mask0 == has_negative) ||
                       (mask1 == has_positive) || (mask1 == has_negative);
  if (both_separated || (kernel.tolerance_int() == Int(0) && either_strict))
    return;

  bool any_zero = combined & has_zero;
  auto is_rep0 = [&](std::size_t i) -> std::pair<bool, bool> {
    auto v_global = Index(poly0.indices()[i]);
    return {Index(fm0[v_global].front()) == face0_id,
            mel0[face0_id][i].is_representative(face0_id)};
  };
  auto is_rep1 = [&](std::size_t i) -> std::pair<bool, bool> {
    auto v_global = Index(poly1.indices()[i]);
    return {Index(fm1[v_global].front()) == face1_id,
            mel1[face1_id][i].is_representative(face1_id)};
  };

  // EF / crossing-EE / crossing-VE (both directions)
  bool both_crossing = (mask0 & has_crossing) == has_crossing &&
                       (mask1 & has_crossing) == has_crossing;
  if ((mask0 & has_crossing) == has_crossing)
    tf::exact::crossing_edges_vs_face(face_buf0, n0, face_buf1, n1, signs0,
                                      tag0, tag1, face0_id, face1_id, is_rep0,
                                      is_rep1, ints, pts, both_crossing,
                                      kernel);
  if ((mask1 & has_crossing) == has_crossing)
    tf::exact::crossing_edges_vs_face(face_buf1, n1, face_buf0, n0, signs1,
                                      tag1, tag0, face1_id, face0_id, is_rep1,
                                      is_rep0, ints, pts, both_crossing,
                                      kernel);
  if (!any_zero)
    return;

  // VV / coplanar-VE / coplanar-EE
  tf::exact::coplanar_primitives(face_buf0, n0, face_buf1, n1, signs0, signs1,
                                 tag0, tag1, face0_id, face1_id, is_rep0,
                                 is_rep1, plane0, plane1, ints, pts, kernel);

  // VF (both directions)
  auto vrep0 = [&](std::size_t i) {
    return Index(fm0[Index(poly0.indices()[i])].front()) == face0_id;
  };
  auto vrep1 = [&](std::size_t j) {
    return Index(fm1[Index(poly1.indices()[j])].front()) == face1_id;
  };
  tf::exact::vertex_face(face_buf0, n0, face_buf1, n1, signs0, tag0, tag1,
                         face0_id, face1_id, vrep0, plane1, ints, pts, kernel);
  tf::exact::vertex_face(face_buf1, n1, face_buf0, n0, signs1, tag1, tag0,
                         face1_id, face0_id, vrep1, plane0, ints, pts, kernel);
}

/// Planes each cached face once (not per candidate pair), then tests pairs.
template <typename Index, typename Int, typename Form0, typename Form1,
          typename MEL0, typename MEL1, typename FM0, typename FM1>
void primitives_process(face_pair_workspace<Index, Int> &ws, const Form0 &form0,
                        const Form1 &form1, int tag0, int tag1,
                        const MEL0 &mel0, const MEL1 &mel1, const FM0 &fm0,
                        const FM1 &fm1,
                        const tf::exact::predicate_kernel<Int> &kernel) {
  auto n0 = ws.n0();
  auto n1 = ws.n1();
  ws.fp0.allocate(n0);
  for (std::size_t i = 0; i < n0; ++i)
    ws.fp0[i] = tf::exact::make_face_plane(ws.face0(i));
  ws.fp1.allocate(n1);
  for (std::size_t j = 0; j < n1; ++j)
    ws.fp1[j] = tf::exact::make_face_plane(ws.face1(j));
  for (std::size_t i = 0; i < n0; ++i)
    for (std::size_t j = 0; j < n1; ++j) {
      if (!tf::intersects(ws.ibox0[i], ws.ibox1[j]))
        continue;
      primitives_polygon_pair_prepped(
          tf::tag_id(Index(ws.ids0[i]), form0[ws.ids0[i]]),
          tf::tag_id(Index(ws.ids1[j]), form1[ws.ids1[j]]), tag0, tag1, mel0,
          mel1, fm0, fm1, ws.face0(i), ws.fp0[i], ws.face1(j), ws.fp1[j],
          ws.intersections, ws.points, kernel);
    }
}

} // namespace tf::intersect
