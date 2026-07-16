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
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/dot_sign.hpp"
#include "../../exact/meta.hpp"
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
/// For each NM edge with fan of `K` occurrences:
///   1. Reject `K < 2`, degenerate original faces, and fans whose
///      member planes do not all contain one line (snapped-merged
///      multi-plane fans) by leaving `is_valid[k] = 0`.
///   2. CCW angular sort of `nm_edge_faces[k]` around the fan's
///      carrier line.
///   3. Rotate so the smallest component label is at position 0.
///   4. Fill the set-canonical key in `id_sorted_view[k]` — the
///      multiset of incident component labels sorted by id.
///
/// Mutates: `nm_edge_faces.data_buffer()` (radial sort, with the
///          per-occurrence direction bits in lockstep).
/// Writes:  `id_sorted_view` (set key), `is_valid`.
///
/// The sort never reads a snapped coordinate. Every decision is a sign
/// predicate on ORIGINAL face normals: an intersection edge lies in
/// every incident original plane, so all wedge normals are exactly
/// perpendicular to the carrier `d = n_f x n_g` (an algebraic identity
/// for the two defining planes, verified exactly for the rest). Hence
/// `n_a x n_b` is parallel to `d`, and every angular comparison is the
/// sign of one cross-product component against `d`'s; the two
/// carrier-degree signs run limb-split so intermediates stay `T2`.
/// Two wedges tie iff their original planes coincide,
/// i.e. exactly the coplanarity the original predicates already
/// decided; rounding can neither create nor destroy an ordering.
///
/// The occurrence's wedge vector is `s * n` with `s` from its
/// per-occurrence direction bit (the loop connection structure). The
/// convention fixes the ring only up to a mirror per fan; a mirrored
/// ring has the same cyclic adjacency, so the emitted sector merges
/// are invariant, and path alignment keeps the choice consistent
/// across each (path, set) bucket for the majority vote.
///
/// @tparam Int         Exact integer type for predicate intermediates.
/// @tparam GetPoint    `(vertex_t v, Index tag) -> point<Int, 3>`.
/// @tparam ApplyToFace `(int tag, int object, callable) -> void` —
///                     callable receives the original-face vertex range.
template <typename Int, typename Index, typename Index1, typename Edges,
          typename Faces, typename DirsView, typename IdView,
          typename LabelsView, typename GetPoint, typename ApplyToFace>
void canonicalize_nm_edges(const tf::arrangement_graph<Index> &,
                           const tf::face_cuts<Index, Index1> &fc,
                           Edges &nm_edges, Faces &nm_edge_faces,
                           const DirsView &dirs_view,
                           const IdView &id_sorted_view,
                           const LabelsView &labels_view,
                           tf::buffer<char> &is_valid, GetPoint get_point,
                           ApplyToFace apply_to_face) {
  using vertex_t = typename tf::cut::non_manifold_edge_fans<Index>::vertex_t;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using nvec = std::array<T2, 3>;

  auto descs = fc.descriptors();

  // Original-plane normal from the face's winding — the same T1-diff /
  // T2-cross form as tf::exact::make_face_plane.
  auto face_normal = [&](Index loop_id) -> nvec {
    const auto desc = descs[loop_id];
    nvec n{T2(0), T2(0), T2(0)};
    apply_to_face(desc.tag, desc.object, [&](const auto &face) {
      auto mk = [](auto id) {
        return vertex_t{tf::intersect::graph::vertex_source::original,
                        Index(id), {}};
      };
      const auto p0 = get_point(mk(face[0]), desc.tag);
      const Index size = Index(face.size());
      T1 e0x = 0, e0y = 0, e0z = 0;
      Index i = 1;
      for (; i < size; ++i) {
        const auto p = get_point(mk(face[i]), desc.tag);
        e0x = T1(p[0]) - p0[0];
        e0y = T1(p[1]) - p0[1];
        e0z = T1(p[2]) - p0[2];
        if (e0x != 0 || e0y != 0 || e0z != 0)
          break;
      }
      for (++i; i < size; ++i) {
        const auto p = get_point(mk(face[i]), desc.tag);
        T1 e1x = T1(p[0]) - p0[0];
        T1 e1y = T1(p[1]) - p0[1];
        T1 e1z = T1(p[2]) - p0[2];
        n[0] = T2(e0y) * e1z - T2(e0z) * e1y;
        n[1] = T2(e0z) * e1x - T2(e0x) * e1z;
        n[2] = T2(e0x) * e1y - T2(e0y) * e1x;
        if (n[0] != 0 || n[1] != 0 || n[2] != 0)
          return;
      }
    });
    return n;
  };

  auto sign_t2 = [](const T2 &v) -> int {
    return (v > 0) ? 1 : (v < 0) ? -1 : 0;
  };
  auto cross3 = [](const nvec &a, const nvec &b) -> nvec {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
  };
  auto dot_sign_t2 = [&](const nvec &a, const nvec &b) -> int {
    return sign_t2(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
  };
  auto is_zero = [](const nvec &v) -> bool {
    return v[0] == 0 && v[1] == 0 && v[2] == 0;
  };

  // ---- Pass 1: per (NM edge, occurrence) wedge normals. --------------
  tf::buffer<nvec> wedges;
  wedges.allocate(nm_edge_faces.data_buffer().size());
  auto wedges_view =
      tf::make_offset_block_range(nm_edge_faces.offsets_buffer(), wedges);

  tf::parallel_for_each(
      tf::zip(nm_edge_faces, dirs_view, wedges_view), [&](auto t) {
        auto &&[face_block, dir_block, wedge_block] = t;
        if (face_block.size() < 2)
          return;
        for (auto &&[loop_id, dir, w] :
             tf::zip(face_block, dir_block, wedge_block)) {
          w = face_normal(loop_id);
          if (!dir)
            for (int c = 0; c < 3; ++c)
              w[c] = -w[c];
        }
      });

  // ---- Pass 2: validity + radial sort on wedge normals. --------------
  tf::parallel_for_each(
      tf::zip(nm_edges, nm_edge_faces, dirs_view, wedges_view, labels_view,
              id_sorted_view, is_valid),
      [&](auto t) {
        auto &&[edge, face_block, dir_block, wedge_block, label_block,
                id_block, valid] = t;
        const Index K = Index(face_block.size());
        if (K < 2)
          return;
        for (auto &&w : wedge_block)
          if (w[0] == 0 && w[1] == 0 && w[2] == 0)
            return; // degenerate original face

        // Carrier line direction: cross of the first two distinct
        // member planes. All-parallel fans are coplanar packs — two
        // antipodal wedge classes, ordered below without a carrier.
        nvec d{T2(0), T2(0), T2(0)};
        for (Index r = 1; r < K; ++r) {
          d = cross3(wedge_block[0], wedge_block[r]);
          if (!is_zero(d))
            break;
        }

        // The side convention downstream assumes the ring is CCW as
        // seen along (v0 -> v1), so the carrier's sign comes from the
        // stored edge — the one snapped-data sign, valid while the
        // edge outmeasures the snap error.
        if (!is_zero(d)) {
          const auto pa = get_point(edge[0], descs[face_block[0]].tag);
          const auto pb = get_point(edge[1], descs[face_block[0]].tag);
          const nvec delta{T2(T1(pb[0]) - pa[0]), T2(T1(pb[1]) - pa[1]),
                           T2(T1(pb[2]) - pa[2])};
          if (tf::exact::dot_sign(d, delta) < 0)
            for (int c = 0; c < 3; ++c)
              d[c] = -d[c];
        }

        if (is_zero(d)) {
          // Coplanar pack: order by antipodal class vs member 0, then
          // by (loop, dir) — deterministic, ties are true coplanarity.
          const nvec ref = wedge_block[0];
          auto zipped = tf::zip(face_block, dir_block, wedge_block);
          std::sort(zipped.begin(), zipped.end(), [&](auto a, auto b) {
            auto &&[fa, da, wa] = a;
            auto &&[fb, db, wb] = b;
            const int ca = dot_sign_t2(wa, ref);
            const int cb = dot_sign_t2(wb, ref);
            if (ca != cb)
              return ca > cb;
            if (fa != fb)
              return fa < fb;
            return da < db;
          });
        } else {
          // Every member plane must contain the carrier line: exact
          // perpendicularity. A snapped-merged multi-plane fan fails
          // here and stays invalid rather than getting a fake order.
          for (auto &&w : wedge_block)
            if (tf::exact::dot_sign(d, w) != 0)
              return;

          // Anchor component of d for axis-sign extraction: n_a x n_b
          // is parallel to d for perpendicular wedges, so its sign
          // along d is the sign of one (nonzero-anchored) component.
          int k_anchor = 0;
          for (int c = 1; c < 3; ++c)
            if ((d[c] < 0 ? -d[c] : d[c]) >
                (d[k_anchor] < 0 ? -d[k_anchor] : d[k_anchor]))
              k_anchor = c;
          const int d_sign = sign_t2(d[k_anchor]);

          auto ccw = [&](const nvec &a, const nvec &b) -> int {
            const nvec c = cross3(a, b);
            return sign_t2(c[k_anchor]) * d_sign;
          };

          const nvec ref = wedge_block[0];
          // Angular class vs the reference ray: 0 = on the ray,
          // 1 = strictly CCW half, 2 = opposite ray, 3 = CW half.
          auto angle_class = [&](const nvec &w) -> int {
            const int c = ccw(ref, w);
            if (c > 0)
              return 1;
            if (c < 0)
              return 3;
            return dot_sign_t2(ref, w) > 0 ? 0 : 2;
          };

          auto zipped = tf::zip(face_block, dir_block, wedge_block);
          std::sort(zipped.begin(), zipped.end(), [&](auto a, auto b) {
            auto &&[fa, da, wa] = a;
            auto &&[fb, db, wb] = b;
            const int ka = angle_class(wa);
            const int kb = angle_class(wb);
            if (ka != kb)
              return ka < kb;
            if (ka == 1 || ka == 3) {
              const int c = ccw(wa, wb);
              if (c != 0)
                return c > 0;
            }
            // Same exact direction — same original plane (true
            // coplanarity); deterministic tie-break.
            if (fa != fb)
              return fa < fb;
            return da < db;
          });
        }

        // Rotate so the smallest component label sits at position 0;
        // occurrence directions travel with their entries.
        auto min_it = std::min_element(label_block.begin(), label_block.end());
        auto rot = min_it - label_block.begin();
        std::rotate(face_block.begin(), face_block.begin() + rot,
                    face_block.end());
        std::rotate(dir_block.begin(), dir_block.begin() + rot,
                    dir_block.end());

        // Set-canonical key: multiset of component labels sorted by id.
        std::copy(label_block.begin(), label_block.end(), id_block.begin());
        std::sort(id_block.begin(), id_block.end());
        valid = 1;
      });
}

} // namespace tf::cut
