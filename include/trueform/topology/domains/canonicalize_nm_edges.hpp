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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient3d.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Pass 1 of @ref tf::make_domain_labels: per NM edge, build the
/// canonical radial ordering of its incident faces.
///
/// For each NM edge `(i, j)`:
///   1. Reject `K < 2`, missing third vertices, and edge-collinear
///      thirds (would-be degenerate cross product in int). Mark the
///      reason in `skip_reason`; leave `is_valid[k] = 0`.
///   2. CCW angular sort of `nm_edge_faces[k]` around the directed
///      axis using exact orient3d.
///   3. Rotate so the smallest fragment label is at position 0.
///   4. Lex-direction pick: compare the label sequence forward vs
///      backward; if backward is lex-smaller, reverse the face block
///      (keeping position 0 fixed) and swap the edge axis in place.
///      This forces a globally consistent direction within a chain
///      of NM edges.
///   5. Fill the set-canonical key in `id_sorted_view[k]` — the
///      multiset of incident fragment labels sorted by id, used as
///      the primary key in Phase B.
///
/// Mutates: `nm_edge_faces.data_buffer()` (radial sort), `nm_edges` (axis swap).
/// Writes:  `id_sorted_view` (set key), `areas_view` (diagnostic),
///          `is_valid`, `skip_reason`.
///
/// @tparam Int Exact integer type for predicate intermediates.
/// @tparam Polygons Polygons range type.
/// @tparam FragLabels Manifold-edge connected-component labels.
/// @tparam Edges Range of NM edges (mutable; axis is swapped).
/// @tparam Faces Mutable offset-block range of incident faces per NM edge.
/// @tparam IdView, AreaView, LabelsView Per-edge block ranges parallel to Faces.
/// @tparam GetPoint Callable mapping vertex id to `tf::point<Int, 3>`.
template <typename Int, typename Polygons, typename FragLabels, typename Edges,
          typename Faces, typename IdView, typename AreaView,
          typename LabelsView, typename GetPoint>
void canonicalize_nm_edges(const Polygons &polygons,
                           const FragLabels &fragment_labels, Edges &nm_edges,
                           Faces &nm_edge_faces, const IdView &id_sorted_view,
                           const AreaView &areas_view,
                           const LabelsView &labels_view,
                           tf::buffer<char> &is_valid,
                           tf::buffer<char> &skip_reason, GetPoint get_point) {
  using Index = std::decay_t<decltype(fragment_labels.labels[0])>;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  /// CCW angular comparator around directed edge axis (i -> j) with
  /// a chosen pivot vertex. Uses exact orient3d_value only; mirrors
  /// the 2D polar-sort pattern in tf::planar_graph_regions.
  auto cmp_around_edge = [&](Index i, Index j, Index pivot) {
    auto pi = get_point(i);
    auto pj = get_point(j);
    auto pp = get_point(pivot);

    auto normal = [](const auto &p1, const auto &p2, const auto &p3) {
      T1 u0 = T1(p2[0]) - p1[0], u1 = T1(p2[1]) - p1[1],
         u2 = T1(p2[2]) - p1[2];
      T1 v0 = T1(p3[0]) - p1[0], v1 = T1(p3[1]) - p1[1],
         v2 = T1(p3[2]) - p1[2];
      return std::array<T2, 3>{T2(u1) * v2 - T2(u2) * v1,
                               T2(u2) * v0 - T2(u0) * v2,
                               T2(u0) * v1 - T2(u1) * v0};
    };
    auto np = normal(pi, pj, pp);

    return [=, &get_point](Index a, Index b) -> bool {
      if (a == pivot)
        return b != pivot;
      if (b == pivot)
        return false;

      auto pa = get_point(a);
      auto pb = get_point(b);

      auto sign = [](T2 v) -> int { return (v > 0) ? 1 : (v < 0) ? -1 : 0; };

      int ya = sign(tf::exact::orient3d_value<Int>(pi, pj, pp, pa));
      int yb = sign(tf::exact::orient3d_value<Int>(pi, pj, pp, pb));

      auto half = [&](int y, const auto &pk) -> bool {
        if (y != 0)
          return y > 0;
        auto nk = normal(pi, pj, pk);
        for (int d = 0; d < 3; ++d) {
          if (np[d] != 0)
            return (nk[d] > 0) == (np[d] > 0);
        }
        return true;
      };

      bool ha = half(ya, pa);
      bool hb = half(yb, pb);
      if (ha != hb)
        return ha > hb;

      int s = sign(tf::exact::orient3d_value<Int>(pi, pj, pa, pb));
      if (s != 0)
        return s > 0;

      return a < b;
    };
  };

  tf::parallel_for_each(
      tf::zip(nm_edges, nm_edge_faces, labels_view, id_sorted_view, areas_view,
              is_valid, skip_reason),
      [&](auto t) {
        auto &&[edge, face_block, label_block, id_block, area_block, valid,
                reason] = t;
        Index K = Index(face_block.size());
        if (K < 2) {
          reason = 1;
          return;
        }

        Index i = edge[0];
        Index j = edge[1];

        auto third_of = [&](Index face_id) -> Index {
          const auto &face = polygons.faces()[face_id];
          for (auto v : face)
            if (v != i && v != j)
              return v;
          return Index(-1);
        };

        auto pi = get_point(i);
        auto pj = get_point(j);
        T1 e0 = T1(pj[0]) - pi[0], e1 = T1(pj[1]) - pi[1],
           e2 = T1(pj[2]) - pi[2];
        for (Index r = 0; r < K; ++r) {
          Index third = third_of(face_block[r]);
          if (third == Index(-1)) {
            reason = 2;
            return;
          }
          auto pk = get_point(third);
          T1 v0 = T1(pk[0]) - pi[0], v1 = T1(pk[1]) - pi[1],
             v2 = T1(pk[2]) - pi[2];
          T2 c0 = T2(e1) * v2 - T2(e2) * v1;
          T2 c1 = T2(e2) * v0 - T2(e0) * v2;
          T2 c2 = T2(e0) * v1 - T2(e1) * v0;
          if (c0 == 0 && c1 == 0 && c2 == 0) {
            reason = 3;
            return;
          }
        }

        Index pivot = third_of(face_block[0]);
        auto cmp = cmp_around_edge(i, j, pivot);
        std::sort(face_block.begin(), face_block.end(),
                  [&](Index fa, Index fb) {
                    return cmp(third_of(fa), third_of(fb));
                  });

        Index min_pos = 0;
        Index min_label = fragment_labels.labels[face_block[0]];
        for (Index r = 1; r < K; ++r) {
          Index lab = fragment_labels.labels[face_block[r]];
          if (lab < min_label) {
            min_label = lab;
            min_pos = r;
          }
        }
        std::rotate(face_block.begin(), face_block.begin() + min_pos,
                    face_block.end());

        for (Index r = 0; r < K; ++r) {
          Index fid = face_block[r];
          const auto &face = polygons.faces()[fid];
          auto p0 = polygons.points()[face[0]];
          auto p1 = polygons.points()[face[1]];
          auto p2 = polygons.points()[face[2]];
          double e0x = double(p1[0]) - p0[0], e0y = double(p1[1]) - p0[1],
                 e0z = double(p1[2]) - p0[2];
          double e1x = double(p2[0]) - p0[0], e1y = double(p2[1]) - p0[1],
                 e1z = double(p2[2]) - p0[2];
          double cx = e0y * e1z - e0z * e1y;
          double cy = e0z * e1x - e0x * e1z;
          double cz = e0x * e1y - e0y * e1x;
          area_block[r] = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
        }

        bool prefer_backward = false;
        for (Index r = 1; r < K; ++r) {
          Index f = label_block[r];
          Index b = label_block[K - r];
          if (f != b) {
            prefer_backward = (b < f);
            break;
          }
        }
        if (prefer_backward) {
          std::reverse(face_block.begin() + 1, face_block.end());
          std::reverse(area_block.begin() + 1, area_block.end());
          std::swap(edge[0], edge[1]);
        }

        std::copy(label_block.begin(), label_block.end(), id_block.begin());
        std::sort(id_block.begin(), id_block.end());
        valid = 1;
      });
}

} // namespace tf::topology::domains
