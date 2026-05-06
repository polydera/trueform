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
#include "../core/algorithm/circular_increment.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/make_equivalence_class_map.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/array_hash.hpp"
#include "../core/buffer.hpp"
#include "../core/complete.hpp"
#include "../core/hash_set.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/sequence_range.hpp"
#include "../exact/int64.hpp"
#include "../exact/meta.hpp"
#include "../exact/orient3d.hpp"
#include "../exact/pt_converter.hpp"
#include "./connected_component_labels.hpp"
#include "./directed_edge_id_in_face.hpp"
#include "./make_face_membership.hpp"
#include "./make_manifold_edge_connected_component_labels.hpp"
#include "./make_manifold_edge_link.hpp"
#include "./non_manifold_edges.hpp"
#include "./policy/face_membership.hpp"
#include "./policy/manifold_edge_link.hpp"
#include <array>
#include <utility>

namespace tf {

/// @ingroup topology_components
/// @brief Per-face per-side domain labels of a non-manifold polygon mesh.
///
/// A non-manifold surface mesh bounds multiple 3D regions ("domains").
/// Each face has two sides; each side bounds one domain. Returns a
/// @ref tf::connected_component_labels with `labels` of size
/// `2 * n_faces` (entry `2*f + s` is the domain id on side `s` of face
/// `f`) and `n_components` set to the total number of domains.
///
/// Algorithm: reformulation of Bohm & Runge 2025 (Computer-Aided Design
/// 180, art. 103824). Manifold-edge connected components are domain
/// fragments by construction; non-manifold edges decide how fragments
/// glue into domains via an angular sort using exact predicates.
///
/// @pre Faces must be consistently oriented within each manifold-edge
///      connected component (e.g. arrangement output, or call
///      @ref tf::orient_faces_consistently first). The per-face "side
///      0 / side 1" mapping is derived from local edge direction;
///      consistent winding makes it coherent across a fragment without
///      an explicit orientation flood.
///
/// @tparam Int Exact integer type for the angular comparator. Predicate
///         intermediates use `tf::exact::meta<Int>`. Default
///         `tf::exact::int64`.
/// @tparam Policy The polygons policy.
/// @param polygons The polygons range.
template <typename Int = tf::exact::int64, typename Policy>
auto make_domain_labels(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;

  if constexpr (!tf::has_face_membership_policy<Policy>) {
    auto fm = tf::make_face_membership(polygons);
    return make_domain_labels<Int>(polygons | tf::tag(fm));
  } else if constexpr (!tf::has_manifold_edge_link_policy<Policy>) {
    auto mel = tf::make_manifold_edge_link(polygons);
    return make_domain_labels<Int>(polygons | tf::tag(mel));
  } else {
    auto fragment_labels =
        tf::make_manifold_edge_connected_component_labels(polygons);
    auto [nm_edges, nm_edge_faces] =
        tf::make_non_manifold_edges(polygons, tf::complete);

    // Build a get_point(v) callable. If the input mesh is already in
    // int coordinates, this is the identity (no roundtrip). Otherwise
    // we build a float→int converter from the AABB. Same pattern as
    // hole_patcher / planar_graph_regions.
    auto get_point = [&]() {
      using CoordT = tf::coordinate_type<Policy>;
      if constexpr (std::is_integral_v<CoordT>) {
        return [&polygons](Index v) -> tf::point<Int, 3> {
          auto p = polygons.points()[v];
          return tf::point<Int, 3>{Int(p[0]), Int(p[1]), Int(p[2])};
        };
      } else {
        auto conv = tf::exact::make_pt_converter<Int>(polygons);
        return [conv, &polygons](Index v) -> tf::point<Int, 3> {
          return conv(polygons.points()[v]);
        };
      }
    }();

    /// CCW angular comparator around directed edge axis (i -> j) with
    /// a chosen pivot vertex. Uses exact orient3d_value only; mirrors
    /// the 2D polar-sort pattern in tf::planar_graph_regions.
    auto cmp_around_edge = [&](Index i, Index j, Index pivot) {
      using T1 = typename tf::exact::meta<Int>::T1;
      using T2 = typename tf::exact::meta<Int>::T2;

      auto pi = get_point(i);
      auto pj = get_point(j);
      auto pp = get_point(pivot);

      // Normal of pivot face (defines the 0° plane and direction)
      auto normal = [](const auto& p1, const auto& p2, const auto& p3) {
        T1 u0 = T1(p2[0]) - p1[0], u1 = T1(p2[1]) - p1[1], u2 = T1(p2[2]) - p1[2];
        T1 v0 = T1(p3[0]) - p1[0], v1 = T1(p3[1]) - p1[1], v2 = T1(p3[2]) - p1[2];
        return std::array<T2, 3>{
          T2(u1)*v2 - T2(u2)*v1,
          T2(u2)*v0 - T2(u0)*v2,
          T2(u0)*v1 - T2(u1)*v0
        };
      };
      auto np = normal(pi, pj, pp);

      return [=, &get_point](Index a, Index b) -> bool {
        if (a == pivot) return b != pivot;
        if (b == pivot) return false;

        auto pa = get_point(a);
        auto pb = get_point(b);

        auto sign = [](T2 v) -> int { return (v > 0) ? 1 : (v < 0) ? -1 : 0; };

        // y is orientation relative to pivot plane
        int ya = sign(tf::exact::orient3d_value<Int>(pi, pj, pp, pa));
        int yb = sign(tf::exact::orient3d_value<Int>(pi, pj, pp, pb));

        // x is "same side of edge" check for coplanar faces
        auto half = [&](int y, const auto& pk) -> bool {
          if (y != 0) return y > 0;
          auto nk = normal(pi, pj, pk);
          // Dot product of normals sign. Since collinear, any non-zero component works.
          for (int d = 0; d < 3; ++d) {
            if (np[d] != 0) return (nk[d] > 0) == (np[d] > 0);
          }
          return true;
        };

        bool ha = half(ya, pa);
        bool hb = half(yb, pb);
        if (ha != hb) return ha > hb;

        // Relative orientation
        int s = sign(tf::exact::orient3d_value<Int>(pi, pj, pa, pb));
        if (s != 0) return s > 0;

        return a < b;
      };
    };

    // Per-NM-edge: sort incident faces CCW, walk cyclically, emit
    // fragment-side merge pairs.
    //
    // NOTE: input is assumed clean — no duplicate (coincident,
    // same-winding) triangles. Duplicates will produce extra trivial
    // 1-face domains in the output (one per orphaned fragment-side
    // node). Clean the mesh upstream if you want a domain count that
    // matches the underlying surface.
    struct local_state_t {
      std::vector<std::pair<Index, Index>> entries;
      tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>> seen;
    };
    tf::buffer<std::array<Index, 2>> merges;
    tf::generic_generate(
        tf::enumerate(nm_edges), merges, local_state_t{},
        [&, &nm_edge_faces = nm_edge_faces](const auto &pair, auto &out_buffer,
                                            local_state_t &state) {
          const auto &[k, edge] = pair;
          Index i = edge[0];
          Index j = edge[1];
          auto incident = nm_edge_faces[k];

          state.entries.clear();
          using T1 = typename tf::exact::meta<Int>::T1;
          using T2 = typename tf::exact::meta<Int>::T2;
          auto pi = get_point(i);
          auto pj = get_point(j);
          T1 e0 = T1(pj[0]) - pi[0], e1 = T1(pj[1]) - pi[1],
             e2 = T1(pj[2]) - pi[2];
          bool degenerate = false;
          for (auto face_id : incident) {
            const auto &face = polygons.faces()[face_id];
            Index third = Index(-1);
            for (auto v : face)
              if (v != i && v != j) {
                third = v;
                break;
              }
            if (third == Index(-1)) {
              degenerate = true;
              break;
            }
            auto pk = get_point(third);
            T1 v0 = T1(pk[0]) - pi[0], v1 = T1(pk[1]) - pi[1],
               v2 = T1(pk[2]) - pi[2];
            T2 c0 = T2(e1) * v2 - T2(e2) * v1;
            T2 c1 = T2(e2) * v0 - T2(e0) * v2;
            T2 c2 = T2(e0) * v1 - T2(e1) * v0;
            if (c0 == 0 && c1 == 0 && c2 == 0) {
              degenerate = true;
              break;
            }
            state.entries.push_back({face_id, third});
          }
          if (degenerate)
            return;

          Index K = Index(state.entries.size());
          if (K < 2)
            return;

          Index pivot_v = state.entries[0].second;
          auto cmp = cmp_around_edge(i, j, pivot_v);
          std::sort(state.entries.begin(), state.entries.end(),
                    [&cmp](const auto &a, const auto &b) {
                      return cmp(a.second, b.second);
                    });

          for (Index r = 0; r < K; ++r) {
            Index Fa = state.entries[r].first;
            Index Fb = state.entries[tf::circular_increment(r, K)].first;
            const auto &face_a = polygons.faces()[Fa];
            const auto &face_b = polygons.faces()[Fb];

            // Side that has oriented edge (i, j) is Side 0.
            // Moving CCW from A to B: merge A's CCW side (0) with B's CW side (1).
            Index sa = tf::directed_edge_id_in_face(i, j, face_a) ==
                               Index(face_a.size())
                           ? 1
                           : 0;
            Index sb = tf::directed_edge_id_in_face(i, j, face_b) ==
                               Index(face_b.size())
                           ? 1
                           : 0;
            sb ^= 1;

            Index node_a = 2 * fragment_labels.labels[Fa] + sa;
            Index node_b = 2 * fragment_labels.labels[Fb] + sb;
            std::array<Index, 2> p = {std::min(node_a, node_b),
                                      std::max(node_a, node_b)};
            if (state.seen.insert(p).second)
              out_buffer.push_back(p);
          }
        });

    // Union-find on the fragment-side graph: each connected component
    // is one domain.
    tf::buffer<Index> domain_of_side;
    domain_of_side.allocate(2 * fragment_labels.n_components);
    auto n_domains =
        tf::make_dense_equivalence_class_map(merges, domain_of_side);

    // Lift to per-face per-side labels: labels[2f + s] = domain on
    // side s of face f.
    Index n_faces = Index(polygons.size());
    tf::connected_component_labels<Index> out;
    out.labels.allocate(2 * n_faces);
    out.n_components = Index(n_domains);
    tf::parallel_for_each(tf::make_sequence_range(n_faces), [&](Index f) {
      Index frag = fragment_labels.labels[f];
      out.labels[2 * f + 0] = domain_of_side[2 * frag + 0];
      out.labels[2 * f + 1] = domain_of_side[2 * frag + 1];
    });
    return out;
  }
}

} // namespace tf
