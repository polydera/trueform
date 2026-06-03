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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/projection_axes.hpp"
#include "../arrangement_graph.hpp"
#include "../face_cuts.hpp"
#include "./make_non_manifold_edge_fans.hpp"
#include <algorithm>
#include <array>

namespace tf::cut {

/// @ingroup cut
/// @brief Per NM edge, build the canonical radial ordering of its
///        incident loop fan.
///
/// Graph-native counterpart to
/// @ref tf::topology::domains::canonicalize_nm_edges, operating on the
/// implicit arrangement carried by @ref tf::arrangement_graph.
///
/// For each NM edge `(vi, vj)` with fan of `K` loop ids:
///   1. Reject `K < 2`, missing third vertices, and edge-collinear
///      thirds (degenerate cross product in int) by leaving
///      `is_valid[k] = 0`.
///   2. CCW angular sort of `nm_edge_faces[k]` around the directed
///      axis using exact orient3d.
///   3. Rotate so the smallest component label is at position 0.
///   4. Fill the set-canonical key in `id_sorted_view[k]` — the
///      multiset of incident component labels sorted by id.
///
/// Mutates: `nm_edge_faces.data_buffer()` (radial sort).
/// Writes:  `id_sorted_view` (set key), `is_valid`.
///
/// Wedge point per fan member: a vertex of its ORIGINAL face triangle
/// projecting LEFT of the directed edge in 2D — mirrors
/// @ref tf::cut::classify_wedge's `find_left_vertex`. Loops are
/// sub-pieces of original faces; their own non-edge vertices include
/// intersection-created vertices that vary across sub-pieces, which
/// would make the radial sort inconsistent. Loop winding inherits the
/// original face's winding, so the direction of `(vi, vj)` in the loop
/// tells us which side of `(pi, pj)` the apex sits on.
///
/// @tparam Int         Exact integer type for predicate intermediates.
/// @tparam GetPoint    `(vertex_t v, Index tag) -> point<Int, 3>`.
/// @tparam ApplyToFace `(int tag, int object, callable) -> void` —
///                     callable receives the original-face vertex range.
template <typename Int, typename Index, typename Index1, typename Edges,
          typename Faces, typename IdView, typename LabelsView,
          typename GetPoint, typename ApplyToFace>
void canonicalize_nm_edges(const tf::arrangement_graph<Index> &,
                           const tf::face_cuts<Index, Index1> &fc,
                           Edges &nm_edges, Faces &nm_edge_faces,
                           const IdView &id_sorted_view,
                           const LabelsView &labels_view,
                           tf::buffer<char> &is_valid, GetPoint get_point,
                           ApplyToFace apply_to_face) {
  using vertex_t = typename tf::cut::non_manifold_edge_fans<Index>::vertex_t;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using pt3 = tf::point<Int, 3>;
  using pt2 = tf::point<Int, 2>;

  auto descs = fc.descriptors();
  auto loops = fc.loops();

  auto edge_in_loop = [&](const auto &loop, const vertex_t &vi,
                          const vertex_t &vj) {
    const Index size = Index(loop.size());
    Index prev = size - 1;
    for (Index i = 0; i < size; prev = i++) {
      const vertex_t lp = loop[prev];
      const vertex_t lc = loop[i];
      if (lp == vi && lc == vj) return true;
      if (lp == vj && lc == vi) return false;
    }
    return true;
  };

  auto left_vertex_of_face = [&](Index loop_id, const pt3 &q0, const pt3 &q1)
      -> pt3 {
    const auto desc = descs[loop_id];
    pt3 result = q0;
    apply_to_face(desc.tag, desc.object, [&](const auto &face) {
      auto mk = [](auto id) {
        return vertex_t{tf::intersect::graph::vertex_source::original,
                        Index(id), {}};
      };
      auto p0 = get_point(mk(face[0]), desc.tag);
      auto p1 = get_point(mk(face[1]), desc.tag);
      auto p2 = get_point(mk(face[2]), desc.tag);
      auto axes = tf::exact::projection_axes(p0, p1, p2);
      pt2 e0_2d{q0[axes.first], q0[axes.second]};
      pt2 e1_2d{q1[axes.first], q1[axes.second]};
      for (auto &&v : face) {
        auto fv = get_point(mk(v), desc.tag);
        pt2 fv_2d{fv[axes.first], fv[axes.second]};
        if (tf::exact::orient2d(e0_2d, e1_2d, fv_2d) > 0) {
          result = fv;
          return;
        }
      }
    });
    return result;
  };

  auto endpoint_points = [&](const auto &face_block, const auto &edge) {
    const Index rep_tag = descs[face_block[0]].tag;
    return std::pair{get_point(edge[0], rep_tag),
                     get_point(edge[1], rep_tag)};
  };

  // ---- Pass 1: precompute per (NM edge, slot) wedge points. ----------
  tf::points_buffer<Int, 3> thirds;
  thirds.allocate(nm_edge_faces.data_buffer().size());
  auto thirds_view =
      tf::make_offset_block_range(nm_edge_faces.offsets_buffer(), thirds);

  tf::parallel_for_each(
      tf::zip(nm_edges, nm_edge_faces, thirds_view), [&](auto t) {
        auto &&[edge, face_block, third_block] = t;
        if (face_block.size() < 2)
          return;
        const vertex_t vi = edge[0];
        const vertex_t vj = edge[1];
        auto endpoints = endpoint_points(face_block, edge);
        const auto &pi = endpoints.first;
        const auto &pj = endpoints.second;
        for (auto &&[loop_id, third] : tf::zip(face_block, third_block)) {
          const bool dir_ij = edge_in_loop(loops[loop_id], vi, vj);
          third = left_vertex_of_face(loop_id, dir_ij ? pi : pj,
                                       dir_ij ? pj : pi);
        }
      });

  // ---- Pass 2: validity + radial sort using precomputed thirds. -----
  auto sign_t2 = [](T2 v) { return (v > 0) ? 1 : (v < 0) ? -1 : 0; };
  auto cross = [](const auto &p1, const auto &p2, const auto &p3) {
    T1 u0 = T1(p2[0]) - p1[0], u1 = T1(p2[1]) - p1[1], u2 = T1(p2[2]) - p1[2];
    T1 v0 = T1(p3[0]) - p1[0], v1 = T1(p3[1]) - p1[1], v2 = T1(p3[2]) - p1[2];
    return std::array<T2, 3>{T2(u1) * v2 - T2(u2) * v1,
                              T2(u2) * v0 - T2(u0) * v2,
                              T2(u0) * v1 - T2(u1) * v0};
  };

  tf::parallel_for_each(
      tf::zip(nm_edges, nm_edge_faces, thirds_view, labels_view,
              id_sorted_view, is_valid),
      [&](auto t) {
        auto &&[edge, face_block, third_block, label_block, id_block, valid] =
            t;
        if (face_block.size() < 2)
          return;
        auto endpoints = endpoint_points(face_block, edge);
        const auto &pi = endpoints.first;
        const auto &pj = endpoints.second;

        // Reject edge-collinear thirds (degenerate cross product).
        for (auto &&pk : third_block) {
          auto c = cross(pi, pj, pk);
          if (c[0] == 0 && c[1] == 0 && c[2] == 0)
            return;
        }

        const pt3 pp = third_block[0];
        const auto np = cross(pi, pj, pp);
        auto cmp_pts = [&](const auto &pa, const auto &pb) {
          int ya = sign_t2(tf::exact::orient3d_value<Int>(pi, pj, pp, pa));
          int yb = sign_t2(tf::exact::orient3d_value<Int>(pi, pj, pp, pb));
          auto half = [&](int y, const auto &pk) {
            if (y != 0) return y > 0;
            auto nk = cross(pi, pj, pk);
            for (int d = 0; d < 3; ++d)
              if (np[d] != 0) return (nk[d] > 0) == (np[d] > 0);
            return true;
          };
          bool ha = half(ya, pa);
          bool hb = half(yb, pb);
          if (ha != hb) return ha > hb;
          int s = sign_t2(tf::exact::orient3d_value<Int>(pi, pj, pa, pb));
          if (s != 0) return s > 0;
          return false;
        };

        // Sort face_block and third_block in lockstep.
        auto zipped = tf::zip(face_block, third_block);
        std::sort(zipped.begin(), zipped.end(), [&](auto a, auto b) {
          auto &&[fa, pa] = a;
          auto &&[fb, pb] = b;
          if (cmp_pts(pa, pb)) return true;
          if (cmp_pts(pb, pa)) return false;
          return fa < fb;
        });

        // Rotate so the smallest component label sits at position 0.
        auto min_it = std::min_element(label_block.begin(), label_block.end());
        std::rotate(face_block.begin(),
                    face_block.begin() + (min_it - label_block.begin()),
                    face_block.end());

        // Set-canonical key: multiset of component labels sorted by id.
        std::copy(label_block.begin(), label_block.end(), id_block.begin());
        std::sort(id_block.begin(), id_block.end());
        valid = 1;
      });
}

} // namespace tf::cut
