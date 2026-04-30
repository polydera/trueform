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
#include "../core/blocked_buffer.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/points.hpp"
#include "../core/views/mapped_range.hpp"
#include "../exact/meta.hpp"
#include "../exact/pt_converter.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/views/sequence_range.hpp"
#include "../exact/incircle.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership.hpp"

namespace tf {

/// @ingroup topology
/// @brief Delaunay refinement of a triangle mesh via Lawson edge flips.
///
/// Takes an existing triangulation (e.g. from ear_cutter) and flips
/// interior edges until the constrained Delaunay property holds.
/// Constrained edges (identified by a user-provided predicate) are never
/// flipped. Non-manifold edges are automatically constrained.
///
/// @tparam Index The index type.
/// @tparam Int The integer type for exact predicates (default: int32).
template <typename Index, typename Int = tf::exact::int32>
class delaunay_flipper {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

public:
  auto clear() -> void {
    _neighbors.clear();
    _stack.clear();
  }

  /// @brief Run Delaunay flips with no additional constraints.
  ///
  /// Only mesh boundary edges (no neighbor) are constrained.
  ///
  /// @param faces Triangle faces (blocked by 3, modified in place).
  /// @param points 2D points for incircle predicates.
  /// @return Number of edges flipped.
  template <typename FacesPolicy, typename PointsPolicy>
  auto flip(tf::faces<FacesPolicy> &faces,
            const tf::points<PointsPolicy> &points) -> Index {
    return flip(faces, points, [](Index, Index) { return false; });
  }

  template <typename FacesPolicy, typename PointsPolicy>
  auto flip(tf::faces<FacesPolicy> &&faces,
            const tf::points<PointsPolicy> &points) -> Index {
    return flip(faces, points);
  }

  template <typename FacesPolicy, typename PointsPolicy,
            typename IsConstrained>
  auto flip(tf::faces<FacesPolicy> &&faces,
            const tf::points<PointsPolicy> &points,
            IsConstrained &&is_constrained) -> Index {
    return flip(faces, points, is_constrained);
  }

  /// @brief Run Delaunay flips with user-provided constraints.
  ///
  /// Constrained edges (where is_constrained(a, b) returns true)
  /// and mesh boundary edges are never flipped.
  ///
  /// @param faces Triangle faces (blocked by 3, modified in place).
  /// @param points 2D points for incircle predicates.
  /// @param is_constrained Callable (Index, Index) -> bool.
  /// @return Number of edges flipped.
  template <typename FacesPolicy, typename PointsPolicy,
            typename IsConstrained>
  auto flip(tf::faces<FacesPolicy> &faces,
            const tf::points<PointsPolicy> &points,
            IsConstrained &&is_constrained) -> Index {
    clear();
    using coord_t = tf::coordinate_type<PointsPolicy>;
    if constexpr (std::is_integral_v<coord_t>) {
      return run(faces, points, is_constrained);
    } else {
      auto conv = tf::exact::make_pt_converter<Int>(points);
      auto int_pts = tf::make_points(tf::make_mapped_range(
          points, [&](const auto &pt) { return conv(pt); }));
      return run(faces, int_pts, is_constrained);
    }
  }

private:
  template <typename FacesPolicy, typename PointsPolicy,
            typename IsConstrained>
  auto run(tf::faces<FacesPolicy> &faces,
           const tf::points<PointsPolicy> &points,
           IsConstrained &&is_constrained) -> Index {
    auto n_tris = static_cast<Index>(faces.size());
    if (n_tris < 2)
      return 0;

    auto n_points = static_cast<Index>(points.size());
    build_neighbors(faces, is_constrained, n_tris, n_points);
    return flip_to_delaunay(faces, points, n_tris);
  }

  // ── Build neighbor array via face_membership ──

  template <typename FacesPolicy, typename IsConstrained>
  auto build_neighbors(const tf::faces<FacesPolicy> &faces,
                       IsConstrained &&is_constrained,
                       Index n_tris, Index n_points) -> void {
    _fm.build(faces, n_points);
    _neighbors.allocate(n_tris);

    tf::parallel_for_each(
        tf::make_sequence_range(n_tris),
        [&](Index t) {
          auto face = faces[t];
          auto &&nb = _neighbors[t];
          for (Index e = 0; e < 3; ++e) {
            Index a = face[e];
            Index b = face[tf::circular_increment(e, Index(3))];

            if (is_constrained(a, b)) {
              nb[e] = Index(-1);
              continue;
            }

            std::array<Index, 2> found;
            auto it = tf::face_edge_neighbors<Index>(
                _fm, faces, t, a, b, found.begin(), found.end());
            auto count = static_cast<Index>(it - found.begin());

            nb[e] = (count == 1) ? found[0] : Index(-1);
          }
        },
        tf::checked);
  }

  // ── Predicates ──

  /// Returns > 0 if (a, b, c) is CCW.
  template <typename PointsPolicy>
  auto orient2d(const tf::points<PointsPolicy> &points,
                Index ia, Index ib, Index ic) const -> T2 {
    auto &&a = points[ia];
    auto &&b = points[ib];
    auto &&c = points[ic];
    return T2(T1(b[0]) - T1(a[0])) * T2(T1(c[1]) - T1(a[1])) -
           T2(T1(b[1]) - T1(a[1])) * T2(T1(c[0]) - T1(a[0]));
  }

  /// Returns > 0 if d is inside circumcircle of (a, b, c) when CCW.
  template <typename PointsPolicy>
  auto in_circle(const tf::points<PointsPolicy> &points,
                 Index ia, Index ib, Index ic, Index id) const -> T2 {
    return tf::exact::incircle<Int>(points[ia], points[ib], points[ic],
                                    points[id]);
  }

  // ── Lawson flip loop (stack-based) ──

  auto find_edge_back(Index t, Index t_from) const -> Index {
    for (Index k = 0; k < 3; ++k)
      if (_neighbors[t][k] == t_from)
        return k;
    return Index(-1);
  }

  auto push_suspect(Index t, Index e) -> void {
    if (t == Index(-1))
      return;
    if (_neighbors[t][e] == Index(-1))
      return;
    _stack.push_back({t, e});
  }

  template <typename FacesPolicy, typename PointsPolicy>
  auto flip_to_delaunay(tf::faces<FacesPolicy> &faces,
                        const tf::points<PointsPolicy> &points,
                        Index n_tris) -> Index {
    // Determine winding: find first non-degenerate triangle
    bool ccw = true;
    for (Index t = 0; t < n_tris; ++t) {
      auto f = faces[t];
      auto o = orient2d(points, f[0], f[1], f[2]);
      if (o != 0) {
        ccw = o > 0;
        break;
      }
    }

    // Seed stack with all interior edges
    _stack.clear();
    _stack.reserve(3 * n_tris);
    for (Index t = 0; t < n_tris; ++t)
      for (Index e = 0; e < 3; ++e)
        if (_neighbors[t][e] != Index(-1) && _neighbors[t][e] > t)
          _stack.push_back({t, e});

    Index total_flipped = 0;

    while (_stack.size() > 0) {
      auto [t, e] = _stack.back();
      _stack.pop_back();

      Index t1 = _neighbors[t][e];
      if (t1 == Index(-1))
        continue;

      Index e1 = find_edge_back(t1, t);
      if (e1 == Index(-1))
        continue;

      auto f0 = faces[t];
      auto f1 = faces[t1];

      Index a = f0[e];
      Index b = f0[tf::circular_increment(e, Index(3))];
      Index c = f0[tf::circular_increment(
          tf::circular_increment(e, Index(3)), Index(3))];
      Index d = f1[tf::circular_increment(
          tf::circular_increment(e1, Index(3)), Index(3))];

      if (c == d)
        continue;

      auto ic = in_circle(points, a, b, c, d);
      if (ccw ? ic <= 0 : ic >= 0)
        continue;

      flip_edge(faces, t, e, t1, e1, c, d);
      ++total_flipped;

      // After flip: t0=(a,d,c) t1=(b,c,d)
      // All 4 outer edges need rechecking:
      push_suspect(t, 0);   // (a→d)
      push_suspect(t, 2);   // (c→a)
      push_suspect(t1, 0);  // (b→c)
      push_suspect(t1, 2);  // (d→b)
    }

    return total_flipped;
  }

  // ── Edge flip ──

  // Flip shared edge between t0 and t1.
  //
  // Before (winding preserved):
  //   t0: a → b → c    (edge e0 is a→b)
  //   t1: b → a → d    (edge e1 is b→a)
  //
  // After:
  //   t0: a → d → c    (keeps a, gains d, keeps c)
  //   t1: b → c → d    (keeps b, gains c, keeps d)
  //
  // Edges:
  //   t0: 0=(a→d) 1=(d→c) 2=(c→a)
  //   t1: 0=(b→c) 1=(c→d) 2=(d→b)
  //   Shared diagonal: t0 edge 1 (d→c) ↔ t1 edge 1 (c→d)
  template <typename FacesPolicy>
  auto flip_edge(tf::faces<FacesPolicy> &faces,
                 Index t0, Index e0, Index t1, Index e1,
                 Index c, Index d) -> void {
    auto &&f0 = faces[t0];
    auto &&f1 = faces[t1];

    Index a = f0[e0];
    Index b = f0[tf::circular_increment(e0, Index(3))];

    // Old neighbors (before overwriting)
    // t0 edges: e0=(a→b), e0+1=(b→c), e0+2=(c→a)
    // t1 edges: e1=(b→a), e1+1=(a→d), e1+2=(d→b)
    Index e0_1 = tf::circular_increment(e0, Index(3));
    Index e0_2 = tf::circular_increment(e0_1, Index(3));
    Index e1_1 = tf::circular_increment(e1, Index(3));
    Index e1_2 = tf::circular_increment(e1_1, Index(3));

    Index nb_bc = _neighbors[t0][e0_1]; // across (b→c)
    Index nb_ca = _neighbors[t0][e0_2]; // across (c→a)
    Index nb_ad = _neighbors[t1][e1_1]; // across (a→d)
    Index nb_db = _neighbors[t1][e1_2]; // across (d→b)

    // Write new triangles
    f0[0] = a; f0[1] = d; f0[2] = c;
    f1[0] = b; f1[1] = c; f1[2] = d;

    // New t0 = (a,d,c): edges 0=(a→d), 1=(d→c), 2=(c→a)
    //   0=(a→d): was t1's (a→d) neighbor
    //   1=(d→c): shared with t1
    //   2=(c→a): was t0's (c→a) neighbor
    _neighbors[t0][0] = nb_ad;
    _neighbors[t0][1] = t1;
    _neighbors[t0][2] = nb_ca;

    // New t1 = (b,c,d): edges 0=(b→c), 1=(c→d), 2=(d→b)
    //   0=(b→c): was t0's (b→c) neighbor
    //   1=(c→d): shared with t0
    //   2=(d→b): was t1's (d→b) neighbor
    _neighbors[t1][0] = nb_bc;
    _neighbors[t1][1] = t0;
    _neighbors[t1][2] = nb_db;

    // Fix back-pointers: external neighbors that pointed to old slots
    if (nb_ad != Index(-1)) {
      for (Index k = 0; k < 3; ++k)
        if (_neighbors[nb_ad][k] == t1) { _neighbors[nb_ad][k] = t0; break; }
    }
    if (nb_bc != Index(-1)) {
      for (Index k = 0; k < 3; ++k)
        if (_neighbors[nb_bc][k] == t0) { _neighbors[nb_bc][k] = t1; break; }
    }
    // nb_ca stays pointing to t0 (same slot index changed but still t0)
    // nb_db stays pointing to t1 (same)
  }

  // ── Data ──

  struct suspect_edge {
    Index tri, edge;
  };

  tf::face_membership<Index> _fm;
  tf::blocked_buffer<Index, 3> _neighbors;
  tf::buffer<suspect_edge> _stack;
};

} // namespace tf
