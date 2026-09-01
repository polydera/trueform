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
#include "../../exact/meta.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/triangle_segment_intersection.hpp"
#include "../../exact/vertex.hpp"
#include "../classify/coplanar_primitives.hpp"
#include "../classify/crossing_edges_vs_face.hpp"
#include "../classify/face_plane_info.hpp"
#include "../classify/intersection_payload.hpp"
#include "../classify/vertex_face.hpp"
#include "./face_pair_search.hpp"
#include "./prepare_face_pair_block.hpp"

#include <array>
#include <optional>
#include <utility>

namespace tf::intersect {

/// First-hit SoS intersection of a segment against a convex face's
/// triangle fan.
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

/// The payload of an edge's SoS pierce through a convex face: the id of
/// the pushed payload, `-1` when the edge misses. `fp` is the face's
/// supporting plane — the payload's second generator — so the caller
/// finds it once per call rather than per crossing.
template <typename Index, typename Int, typename Pts>
auto push_sos_edge_plane_hit(tf::exact::vertex_range<Index, Int> face_verts,
                             const tf::exact::face_plane<Int> &fp,
                             const tf::exact::vertex<Index, Int> &v0,
                             const tf::exact::vertex<Index, Int> &v1, Pts &pts)
    -> Index {
  auto pt = edge_vs_convex_face_sos(face_verts, v0, v1);
  if (!pt)
    return Index(-1);
  Index id = pts.size();
  pts.push_back(
      tf::exact::make_sos_edge_plane_payload<typename Pts::value_type>(
          *pt, face_verts[fp.info.i0].pt, face_verts[fp.info.i1].pt,
          face_verts[fp.info.i2].pt, v0, v1));
  return id;
}

/// The supporting plane an SoS payload needs as its second generator, or
/// a null plane when the payload states no parameter.
template <typename Payload, typename Index, typename Int>
auto sos_edge_plane(tf::exact::vertex_range<Index, Int> face_verts)
    -> tf::exact::face_plane<Int> {
  if constexpr (tf::exact::stores_edge_fractions<Payload, Int, Index>)
    return tf::exact::make_face_plane(face_verts);
  else
    return {};
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
  auto fp = sos_edge_plane<typename Pts::value_type>(face_verts);
  for (std::size_t j = 0; j < n; ++j) {
    if (!edge_is_rep(j))
      continue;
    auto next_j = tf::circular_increment(j, n);
    auto id = push_sos_edge_plane_hit(face_verts, fp, edge_verts[j],
                                      edge_verts[next_j], pts);
    if (id == Index(-1))
      continue;
    auto edge_target = tf::topo_id<Index>{Index(j), tf::topo_type::edge};
    auto face_target = tf::topo_id<Index>{face_id, tf::topo_type::face};
    ints.push_back({short(edge_tag), short(face_tag), edge_id, face_id,
                    edge_target, face_target, id});
  }
}

/// Cross-pair SoS over a prepped workspace leaf pair.
template <typename Index, typename Int, typename Payload, typename MEL0,
          typename MEL1>
void sos_process(face_pair_workspace<Index, Int, Payload> &ws, int tag0,
                 int tag1, const MEL0 &mel0, const MEL1 &mel1) {
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
                        ws.intersections, ws.payloads);
      edges_vs_face_sos(ws.face1(j), id1, ws.face0(i), id0, tag1, tag0, erep1,
                        ws.intersections, ws.payloads);
    }
}

/// A plane's exact orient3d values at another face's vertices. They are
/// the numerator and denominator of a pierce's parameter along its edge,
/// so only the payload that states a pierce as a parameter asks for them,
/// and only once its edge's sign pair has already proved the crossing.
template <typename Index, typename Int>
void compute_plane_values(
    const tf::exact::orient3d_plane<Int> &plane,
    tf::exact::vertex_range<Index, Int> verts,
    tf::small_vector<typename tf::exact::meta<Int>::T2, 16> &out) {
  out.clear();
  for (const auto &v : verts)
    out.push_back(tf::exact::orient3d_plane_value(plane, v.pt));
}

/// Cross-pair primitives classification of two prepped faces (ranges +
/// cached planes). EF / crossing-EE / crossing-VE, then the coplanar
/// family and VF when a zero sign appears.
template <typename Index, typename Int, typename Payload, typename Poly0,
          typename Poly1, typename MEL0, typename MEL1, typename FM0,
          typename FM1>
auto primitives_polygon_pair_prepped(
    face_pair_workspace<Index, Int, Payload> &ws, const Poly0 &poly0,
    const Poly1 &poly1, int tag0, int tag1, const MEL0 &mel0, const MEL1 &mel1,
    const FM0 &fm0, const FM1 &fm1,
    tf::exact::vertex_range<Index, Int> face_buf0,
    const tf::exact::face_plane<Int> &fp0,
    tf::exact::vertex_range<Index, Int> face_buf1,
    const tf::exact::face_plane<Int> &fp1) {
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

  auto &signs0 = ws.signs0;
  auto &signs1 = ws.signs1;
  // An invalid plane states no sign, and zero is what every consumer of
  // these reads as "on the plane" — the degenerate face's answer.
  signs0.assign(n0, 0);
  signs1.assign(n1, 0);
  int mask0 = 0, mask1 = 0;
  // The signs are the one producer of "on the plane" — the separating
  // reject and every coplanarity guard below read them.
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

  // Separating-plane reject: a pure one-sided mask certifies that every
  // point of that face — which lies in its supporting plane — clears the
  // other plane.
  int combined = mask0 | mask1;
  bool both_separated = !(combined & has_zero) &&
                        (mask0 & has_crossing) != has_crossing &&
                        (mask1 & has_crossing) != has_crossing;
  bool either_strict = (mask0 == has_positive) || (mask0 == has_negative) ||
                       (mask1 == has_positive) || (mask1 == has_negative);
  if (both_separated || either_strict)
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
  if ((mask0 & has_crossing) == has_crossing) {
    if constexpr (tf::exact::stores_edge_fractions<Payload, Int, Index>)
      compute_plane_values(fp1.plane, face_buf0, ws.values0);
    tf::exact::crossing_edges_vs_face(
        face_buf0, n0, face_buf1, n1, signs0, ws.values0, tag0, tag1, face0_id,
        face1_id, is_rep0, is_rep1, ws.intersections, ws.payloads,
        both_crossing);
  }
  if ((mask1 & has_crossing) == has_crossing) {
    if constexpr (tf::exact::stores_edge_fractions<Payload, Int, Index>)
      compute_plane_values(fp0.plane, face_buf1, ws.values1);
    tf::exact::crossing_edges_vs_face(
        face_buf1, n1, face_buf0, n0, signs1, ws.values1, tag1, tag0, face1_id,
        face0_id, is_rep1, is_rep0, ws.intersections, ws.payloads,
        both_crossing);
  }
  if (!any_zero)
    return;

  // VV / coplanar-VE / coplanar-EE
  tf::exact::coplanar_primitives(face_buf0, n0, face_buf1, n1, signs0, signs1,
                                 tag0, tag1, face0_id, face1_id, is_rep0,
                                 is_rep1, plane0, plane1, ws.intersections,
                                 ws.payloads);

  // VF (both directions)
  auto vrep0 = [&](std::size_t i) {
    return Index(fm0[Index(poly0.indices()[i])].front()) == face0_id;
  };
  auto vrep1 = [&](std::size_t j) {
    return Index(fm1[Index(poly1.indices()[j])].front()) == face1_id;
  };
  tf::exact::vertex_face(face_buf0, n0, face_buf1, n1, signs0, tag0, tag1,
                         face0_id, face1_id, vrep0, plane1, ws.intersections,
                         ws.payloads);
  tf::exact::vertex_face(face_buf1, n1, face_buf0, n0, signs1, tag1, tag0,
                         face1_id, face0_id, vrep1, plane0, ws.intersections,
                         ws.payloads);

  // COPLANARITY IS THE NAME, AND THE SIGNS ARE ITS ROUTE.
  //
  // A sign says where a point lies RELATIVE TO a plane; it cannot say
  // that two planes ARE one. So the name is the verdict: it is a value,
  // the question has one answer per face, and the classifier and the
  // plane graph ask the same producer for it.
  //
  // But the signs decide WHO IS ASKED. The three corners a carrier's
  // plane stands on lie on that plane exactly, so two carriers wearing
  // one name put each other's support corners on their own plane. A pair
  // that fails this reading can therefore never be name-equal, and never
  // asks. The reading is the sign array the decision tree has already
  // filled, so the average pair pays a comparison it has already made and
  // no carrier is named to answer a question that is already settled.
  //
  // The fact belongs to the PAIR, not to records: representative gating
  // may route contacts through other pairs' calls, so a stamp on this
  // call's emissions can be lost. Collected here, distributed onto the
  // final records after the build.
  const auto support_on_plane = [](const tf::exact::face_plane_info &info,
                                   const auto &signs) {
    return signs[info.i0] == 0 && signs[info.i1] == 0 && signs[info.i2] == 0;
  };
  if (plane0.valid && plane1.valid && support_on_plane(plane0, signs0) &&
      support_on_plane(plane1, signs1) &&
      tf::exact::face_plane_name(fp0, face_buf0) ==
          tf::exact::face_plane_name(fp1, face_buf1))
    ws.coplanar_pairs.push_back({Index(tag0), face0_id, Index(tag1), face1_id});
}

/// Prepares the block's kept pairs and their planes, then tests pairs.
template <typename Index, typename Int, typename Payload, typename Form0,
          typename Form1, typename MEL0, typename MEL1, typename FM0,
          typename FM1>
void primitives_process(face_pair_workspace<Index, Int, Payload> &ws,
                        const Form0 &form0, const Form1 &form1, int tag0,
                        int tag1, const MEL0 &mel0, const MEL1 &mel1,
                        const FM0 &fm0, const FM1 &fm1) {
  auto n0 = ws.n0();
  auto n1 = ws.n1();
  prepare_face_pair_block(ws, false);
  for (std::size_t i = 0; i < n0; ++i)
    for (std::size_t j = 0; j < n1; ++j) {
      if (!ws.pair_kept[i * n1 + j])
        continue;
      primitives_polygon_pair_prepped(
          ws, tf::tag_id(Index(ws.ids0[i]), form0[ws.ids0[i]]),
          tf::tag_id(Index(ws.ids1[j]), form1[ws.ids1[j]]), tag0, tag1, mel0,
          mel1, fm0, fm1, ws.face0(i), ws.fp0[i], ws.face1(j), ws.fp1[j]);
    }
}

} // namespace tf::intersect
