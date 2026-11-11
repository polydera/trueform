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
#include "../../core/config/verify.hpp"
#include "./certify_shared_feature_pair.hpp"
#include "./face_images.hpp"
#include "./face_pair_kernels.hpp"
#include "./keep_face_pair_block.hpp"

namespace tf::intersect {

/// Shared-vertex masks of two faces of one form: `s0[i]` / `s1[j]` mark
/// positions whose vertex id appears in the other face.
template <typename Index, typename Indices0, typename Indices1>
auto compute_shared_masks(const Indices0 &ids0, const Indices1 &ids1,
                          tf::buffer<bool> &s0, tf::buffer<bool> &s1) {
  s0.allocate(ids0.size());
  s1.allocate(ids1.size());
  std::fill(s0.begin(), s0.end(), false);
  std::fill(s1.begin(), s1.end(), false);
  for (decltype(ids0.size()) i = 0; i < ids0.size(); ++i)
    for (decltype(ids1.size()) j = 0; j < ids1.size(); ++j)
      if (Index(ids0[i]) == Index(ids1[j]))
        s0[i] = s1[j] = true;
}

inline auto has_shared_edge(const tf::buffer<bool> &mask) -> bool {
  std::size_t n = mask.size();
  for (std::size_t i = 0; i < n; ++i)
    if (mask[i] && mask[tf::circular_increment(i, n)])
      return true;
  return false;
}

/// Self SoS: edges of one prepped face vs the other's face; skips edges
/// with a shared endpoint (a shared edge does not self-intersect).
template <typename Index, typename Int, typename EdgeIsRep, typename Ints,
          typename Pts>
auto edges_vs_face_sos(tf::exact::vertex_range<Index, Int> edge_verts,
                       Index edge_id,
                       tf::exact::vertex_range<Index, Int> face_verts,
                       Index face_id, int tag, const EdgeIsRep &edge_is_rep,
                       const tf::buffer<bool> &shared, Ints &ints, Pts &pts) {
  auto n = edge_verts.size();
  for (std::size_t j = 0; j < n; ++j) {
    auto next_j = tf::circular_increment(j, n);
    if (shared[j] || shared[next_j])
      continue;
    if (!edge_is_rep(j))
      continue;
    auto id = push_sos_edge_plane_hit(face_verts, edge_verts[j],
                                      edge_verts[next_j], pts);
    if (id == Index(-1))
      continue;
    ints.push_back({short(tag),
                    short(tag),
                    edge_id,
                    face_id,
                    {Index(j), tf::topo_type::edge},
                    {face_id, tf::topo_type::face},
                    id});
  }
}

/// Self SoS over a workspace leaf pair, reading the form's face images.
template <typename Index, typename Int, typename Form, typename MEL>
void self_sos_process(face_pair_workspace<Index, Int> &ws,
                      const face_images<Index, Int> &images, bool is_self,
                      const Form &form, int tag, const MEL &mel) {
  keep_face_pair_block(ws, is_self);
  for (const auto &kept : ws.kept_pairs) {
    const auto f0 = ws.ids0[std::size_t(kept[0])];
    const auto f1 = ws.ids1[std::size_t(kept[1])];
    auto id0 = Index(f0);
    auto id1 = Index(f1);
    compute_shared_masks<Index>(form[f0].indices(), form[f1].indices(),
                                ws.shared0, ws.shared1);
    if (has_shared_edge(ws.shared0))
      continue;
    auto erep0 = [&](std::size_t e) {
      return mel[id0][e].is_representative(id0);
    };
    auto erep1 = [&](std::size_t e) {
      return mel[id1][e].is_representative(id1);
    };
    edges_vs_face_sos(images.face(f0), id0, images.face(f1), id1, tag, erep0,
                      ws.shared0, ws.intersections, ws.payloads);
    edges_vs_face_sos(images.face(f1), id1, images.face(f0), id0, tag, erep1,
                      ws.shared1, ws.intersections, ws.payloads);
  }
}

/// Self pair test over two prepped faces (ranges + cached planes + shared
/// masks). Shared-aware masks, `_self` crossing dedup, both-sided reject.
template <typename Index, typename Int, typename Poly0, typename Poly1,
          typename MEL, typename FM>
auto within_polygon_pair_prepped(face_pair_workspace<Index, Int> &ws,
                                 const Poly0 &poly0, const Poly1 &poly1,
                                 int tag, const MEL &mel, const FM &fm,
                                 tf::exact::vertex_range<Index, Int> face_buf0,
                                 const tf::exact::face_plane<Int> &fp0,
                                 tf::exact::vertex_range<Index, Int> face_buf1,
                                 const tf::exact::face_plane<Int> &fp1) {
  const auto &shared0 = ws.shared0;
  const auto &shared1 = ws.shared1;
  auto face0_id = Index(poly0.id());
  auto face1_id = Index(poly1.id());

  auto n0 = face_buf0.size();
  auto n1 = face_buf1.size();

  const auto plane0 = tf::exact::face_orient3d_plane(fp0, face_buf0);
  const auto plane1 = tf::exact::face_orient3d_plane(fp1, face_buf1);

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
  int mask0_ns = 0, mask1_ns = 0;
  if (fp1.valid()) {
    for (decltype(n0) i = 0; i < n0; ++i) {
      signs0[i] = tf::exact::orient3d_plane_sign(plane1, face_buf0[i].pt);
      auto bit = 1 << (signs0[i] + 1);
      mask0 |= bit;
      if (!shared0[i])
        mask0_ns |= bit;
    }
  }
  if (fp0.valid()) {
    for (decltype(n1) j = 0; j < n1; ++j) {
      signs1[j] = tf::exact::orient3d_plane_sign(plane0, face_buf1[j].pt);
      auto bit = 1 << (signs1[j] + 1);
      mask1 |= bit;
      if (!shared1[j])
        mask1_ns |= bit;
    }
  }

  int combined = mask0 | mask1;
  int combined_ns = mask0_ns | mask1_ns;
  if (!(combined_ns & has_zero) && (mask0_ns & has_crossing) != has_crossing &&
      (mask1_ns & has_crossing) != has_crossing)
    return;

  bool any_zero = combined & has_zero;
  auto is_rep0 = [&](std::size_t i) -> std::pair<bool, bool> {
    auto v_global = poly0.indices()[i];
    return {Index(fm[v_global].front()) == face0_id,
            mel[face0_id][i].is_representative(face0_id)};
  };
  auto is_rep1 = [&](std::size_t i) -> std::pair<bool, bool> {
    auto v_global = poly1.indices()[i];
    return {Index(fm[v_global].front()) == face1_id,
            mel[face1_id][i].is_representative(face1_id)};
  };

  auto shared0_f = [&shared0](std::size_t i) { return shared0[i]; };
  auto shared1_f = [&shared1](std::size_t j) { return shared1[j]; };

  bool both_crossing = (mask0 & has_crossing) == has_crossing &&
                       (mask1 & has_crossing) == has_crossing;
  if ((mask0 & has_crossing) == has_crossing) {
    compute_plane_values(plane1, face_buf0, ws.values0);
    tf::exact::crossing_edges_vs_face_self(
        face_buf0, n0, face_buf1, n1, signs0, ws.values0, tag, tag, face0_id,
        face1_id, is_rep0, is_rep1, ws.intersections, ws.payloads,
        both_crossing);
  }
  if ((mask1 & has_crossing) == has_crossing) {
    compute_plane_values(plane0, face_buf1, ws.values1);
    tf::exact::crossing_edges_vs_face_self(
        face_buf1, n1, face_buf0, n0, signs1, ws.values1, tag, tag, face1_id,
        face0_id, is_rep1, is_rep0, ws.intersections, ws.payloads,
        both_crossing);
  }
  if (!any_zero)
    return;

  const auto info0 = tf::exact::make_face_plane_info(fp0);
  const auto info1 = tf::exact::make_face_plane_info(fp1);
  tf::exact::coplanar_primitives(face_buf0, n0, face_buf1, n1, signs0, signs1,
                                 tag, tag, face0_id, face1_id, is_rep0, is_rep1,
                                 info0, info1, shared0_f, shared1_f,
                                 ws.intersections, ws.payloads);

  auto vrep0 = [&](std::size_t i) {
    return Index(fm[poly0.indices()[i]].front()) == face0_id;
  };
  auto vrep1 = [&](std::size_t j) {
    return Index(fm[poly1.indices()[j]].front()) == face1_id;
  };
  tf::exact::vertex_face(face_buf0, n0, face_buf1, n1, signs0, tag, tag,
                         face0_id, face1_id, vrep0, shared0_f, info1,
                         ws.intersections, ws.payloads);
  tf::exact::vertex_face(face_buf1, n1, face_buf0, n0, signs1, tag, tag,
                         face1_id, face0_id, vrep1, shared1_f, info0,
                         ws.intersections, ws.payloads);

  // A pair whose vertices all read zero both ways lies on one plane — a
  // coincident contact region. The fact belongs to the PAIR, not to an
  // emission: representative gating may route every contact through other
  // pairs' calls, so a stamp on this call's emissions can be lost.
  // Collected here, distributed onto the final records after the build.
  if (mask0 == has_zero && mask1 == has_zero)
    ws.coplanar_pairs.push_back({Index(tag), face0_id, Index(tag), face1_id});
}

/// Per-leaf-pair self primitives logic, reading the form's face images.
template <typename Index, typename Int, typename Form, typename MEL,
          typename FM>
void self_process(face_pair_workspace<Index, Int> &ws,
                  const face_images<Index, Int> &images, bool is_self,
                  const Form &form, int tag, const MEL &mel, const FM &fm) {
  keep_face_pair_block(ws, is_self);
  for (const auto &kept : ws.kept_pairs) {
    const auto f0 = ws.ids0[std::size_t(kept[0])];
    const auto f1 = ws.ids1[std::size_t(kept[1])];
    auto poly0 = tf::tag_id(Index(f0), form[f0]);
    auto poly1 = tf::tag_id(Index(f1), form[f1]);
    const auto face0 = images.face(f0);
    const auto face1 = images.face(f1);
    const auto &fp0 = images.planes[f0];
    const auto &fp1 = images.planes[f1];
    if (certify_shared_feature_pair<Index>(poly0.indices(), poly1.indices(),
                                           face0, fp0, face1, fp1)) {
      if constexpr (tf::core::verification) {
        const auto n_records = ws.intersections.size();
        const auto n_coplanar = ws.coplanar_pairs.size();
        compute_shared_masks<Index>(poly0.indices(), poly1.indices(),
                                    ws.shared0, ws.shared1);
        within_polygon_pair_prepped(ws, poly0, poly1, tag, mel, fm, face0, fp0,
                                    face1, fp1);
        tf::core::verify(ws.intersections.size() == n_records &&
                             ws.coplanar_pairs.size() == n_coplanar,
                         "a certified shared-feature pair states no record");
      }
      continue;
    }
    compute_shared_masks<Index>(poly0.indices(), poly1.indices(), ws.shared0,
                                ws.shared1);
    within_polygon_pair_prepped(ws, poly0, poly1, tag, mel, fm, face0, fp0,
                                face1, fp1);
  }
}

} // namespace tf::intersect
