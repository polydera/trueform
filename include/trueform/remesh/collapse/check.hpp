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

#include <limits>
#include <type_traits>

#include "../../core/constants.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/transformed.hpp"
#include "../../topology/half_edges.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Configurable collapse validity check.
///
/// Iterates the 1-rings of both endpoints and applies the enabled
/// checks to every surviving face. Template bools select checks
/// at compile time — disabled checks have zero cost.
///
/// NormalFlip uses a two-phase accumulated approach: flipped faces
/// are collected, then only rejected if their new normal opposes the
/// average new-surface normal. This is more permissive than a strict
/// per-face flip test.
///
/// @tparam NormalFlip  Reject if a flipped face opposes the average normal.
/// @tparam MinQuality  Reject if worst triangle quality drops below
///                     min(min_quality, worst old quality). See
///                     is_collapse_allowed_dihedral for the [0,1] semantics.
/// @tparam EdgeLength  Reject if longest ring edge increases (or exceeds
///                     max_edge_length_sq when > 0).
template <bool NormalFlip = true, typename Index, typename PointsPolicy,
          typename Real, typename MinQualityT = tf::none_t,
          typename EdgeLengthT = tf::none_t>
auto is_collapse_allowed(const tf::half_edges<Index> &he,
                         const tf::points<PointsPolicy> &points,
                         tf::half_edge_handle<Index> heh,
                         const tf::point<Real, 3> &pt,
                         MinQualityT min_quality = tf::none,
                         EdgeLengthT max_edge_length_sq = tf::none) -> bool {
  constexpr bool Quality = !std::is_same_v<MinQualityT, tf::none_t>;
  constexpr bool EdgeLength = !std::is_same_v<EdgeLengthT, tf::none_t>;
  auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
  auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();

  auto frame = tf::frame_of(points);

  const Real kq = tf::two_over_sqrt_3<Real>;
  Real min_old_q = std::numeric_limits<Real>::max();
  Real min_new_q = std::numeric_limits<Real>::max();
  bool q_active = false;
  Real floor_q = Real(0);
  if constexpr (Quality) {
    Real mq = Real(min_quality);
    q_active = mq >= Real(0);
    floor_q = (mq == Real(0)) ? std::numeric_limits<Real>::max() : mq;
  }
  Real max_old_e2 = 0, max_new_e2 = 0;

  // Accumulated normal flip state
  double sum_nx = 0, sum_ny = 0, sum_nz = 0;
  static constexpr int max_flips = 32;
  Real flip_buf[max_flips * 3];
  int n_flipped = 0;

  // v0's ring
  {
    auto cur = he.rotated(heh);
    if (!cur.is_valid())
      return true;
    while (cur != heh) {
      auto a = he.end_vertex_handle(tf::unsafe, cur).id();
      auto b = he.end_vertex_handle(tf::unsafe, he.next(tf::unsafe, cur)).id();

      if (b != v1) {
        auto ea = points[a] - points[v0];
        auto eb = points[b] - points[v0];
        auto fa = points[a] - pt;
        auto fb = points[b] - pt;

        if constexpr (NormalFlip || Quality) {
          auto old_n = tf::cross(ea, eb);
          auto new_n = tf::cross(fa, fb);

          if constexpr (NormalFlip) {
            sum_nx += double(new_n[0]);
            sum_ny += double(new_n[1]);
            sum_nz += double(new_n[2]);

            if (tf::dot(old_n, new_n) <= Real(0)) {
              if (n_flipped >= max_flips)
                return false;
              flip_buf[n_flipped * 3 + 0] = new_n[0];
              flip_buf[n_flipped * 3 + 1] = new_n[1];
              flip_buf[n_flipped * 3 + 2] = new_n[2];
              ++n_flipped;
            }
          }

          if constexpr (Quality) {
            if (q_active) {
              auto ec = points[b] - points[a];
              Real ec2 = tf::dot(ec, ec);

              Real old_a2 = old_n.length2();
              Real old_me2 = std::max({tf::dot(ea, ea), tf::dot(eb, eb), ec2});
              if (old_me2 > 0) {
                Real qo = kq * std::sqrt(std::max(old_a2, Real(0))) / old_me2;
                min_old_q = std::min(min_old_q, qo);
              }

              Real new_a2 = new_n.length2();
              Real new_me2 = std::max({tf::dot(fa, fa), tf::dot(fb, fb), ec2});
              if (new_me2 > 0) {
                Real qn = kq * std::sqrt(std::max(new_a2, Real(0))) / new_me2;
                min_new_q = std::min(min_new_q, qn);
              }
            }
          }
        }

        if constexpr (EdgeLength) {
          max_old_e2 = std::max({max_old_e2,
                                 tf::transformed(ea, frame).length2(),
                                 tf::transformed(eb, frame).length2()});
          max_new_e2 = std::max({max_new_e2,
                                 tf::transformed(fa, frame).length2(),
                                 tf::transformed(fb, frame).length2()});
        }
      }

      cur = he.rotated(cur);
      if (!cur.is_valid())
        return true;
    }
  }

  // v1's ring
  {
    auto opp = he.opposite(tf::unsafe, heh);
    auto cur = he.rotated(opp);
    if (!cur.is_valid())
      return true;
    while (cur != opp) {
      auto a = he.end_vertex_handle(tf::unsafe, cur).id();
      auto b = he.end_vertex_handle(tf::unsafe, he.next(tf::unsafe, cur)).id();

      if (b != v0) {
        auto ea = points[a] - points[v1];
        auto eb = points[b] - points[v1];
        auto fa = points[a] - pt;
        auto fb = points[b] - pt;

        if constexpr (NormalFlip || Quality) {
          auto old_n = tf::cross(ea, eb);
          auto new_n = tf::cross(fa, fb);

          if constexpr (NormalFlip) {
            sum_nx += double(new_n[0]);
            sum_ny += double(new_n[1]);
            sum_nz += double(new_n[2]);

            if (tf::dot(old_n, new_n) <= Real(0)) {
              if (n_flipped >= max_flips)
                return false;
              flip_buf[n_flipped * 3 + 0] = new_n[0];
              flip_buf[n_flipped * 3 + 1] = new_n[1];
              flip_buf[n_flipped * 3 + 2] = new_n[2];
              ++n_flipped;
            }
          }

          if constexpr (Quality) {
            if (q_active) {
              auto ec = points[b] - points[a];
              Real ec2 = tf::dot(ec, ec);

              Real old_a2 = old_n.length2();
              Real old_me2 = std::max({tf::dot(ea, ea), tf::dot(eb, eb), ec2});
              if (old_me2 > 0) {
                Real qo = kq * std::sqrt(std::max(old_a2, Real(0))) / old_me2;
                min_old_q = std::min(min_old_q, qo);
              }

              Real new_a2 = new_n.length2();
              Real new_me2 = std::max({tf::dot(fa, fa), tf::dot(fb, fb), ec2});
              if (new_me2 > 0) {
                Real qn = kq * std::sqrt(std::max(new_a2, Real(0))) / new_me2;
                min_new_q = std::min(min_new_q, qn);
              }
            }
          }
        }

        if constexpr (EdgeLength) {
          max_old_e2 = std::max({max_old_e2,
                                 tf::transformed(ea, frame).length2(),
                                 tf::transformed(eb, frame).length2()});
          max_new_e2 = std::max({max_new_e2,
                                 tf::transformed(fa, frame).length2(),
                                 tf::transformed(fb, frame).length2()});
        }
      }

      cur = he.rotated(cur);
      if (!cur.is_valid())
        return true;
    }
  }

  if constexpr (NormalFlip) {
    if (n_flipped > 0) {
      double len2 = sum_nx * sum_nx + sum_ny * sum_ny + sum_nz * sum_nz;
      if (len2 < std::numeric_limits<double>::epsilon())
        return false;
      double inv = 1.0 / tf::sqrt(len2);
      double nx = sum_nx * inv, ny = sum_ny * inv, nz = sum_nz * inv;
      for (int i = 0; i < n_flipped; ++i) {
        if (double(flip_buf[i * 3 + 0]) * nx +
                double(flip_buf[i * 3 + 1]) * ny +
                double(flip_buf[i * 3 + 2]) * nz <
            0)
          return false;
      }
    }
  }

  if constexpr (Quality) {
    if (q_active && min_new_q < std::min(floor_q, min_old_q))
      return false;
  }

  if constexpr (EdgeLength) {
    if (max_new_e2 > max_old_e2)
      return false;
    if (max_edge_length_sq > 0 && max_new_e2 > max_edge_length_sq)
      return false;
  }

  return true;
}

/// @ingroup remesh
/// @brief Dihedral angle collapse validity check.
///
/// Builds the vertex link (ordered ring around the collapse point),
/// then checks adjacent new-face pairs for area degeneracy and
/// sharp dihedral creases. Optionally enforces aspect ratio and
/// max edge length.
///
/// @param check_normals When true, additionally reject the collapse if ANY
///                      surviving face's normal inverts (its post-collapse
///                      normal opposes its pre-collapse normal, dot <= 0). This
///                      is CGAL's Bounded_normal_change guard. Unlike the
///                      dihedral test (adjacent NEW faces vs each other), this
///                      compares each face's OLD vs NEW normal, so it catches a
///                      COHERENT inversion of a whole 1-ring fan that the
///                      dihedral test misses. An old face too degenerate for
///                      its own normal to be meaningful defers to the
///                      aggregate of the ring's old normals; a new face whose
///                      orientation sign is below the determinable floor is
///                      rejected. Runtime flag, default false.
/// @tparam MinQualityT  Pass a Real value to enable the triangle-quality
///                      check. The value is min_quality in [0,1] (1 =
///                      equilateral, ->0 = sliver): a collapse is allowed iff
///                      the worst NEW quality >= min(min_quality, worst OLD
///                      quality). So 0 = never worsen, >0 = quality floor (that
///                      still permits matching an already-worse ring, and always
///                      permits an improvement), <0 disables. Default tf::none
///                      disables the check.
/// @tparam EdgeLengthT  Pass a Real value to enable the edge length
///                      check (used as floor for max old edge²).
///                      Default tf::none disables the check.
template <typename Index, typename PointsPolicy, typename Real,
          typename MinQualityT = tf::none_t, typename EdgeLengthT = tf::none_t>
auto is_collapse_allowed_dihedral(
    const tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    tf::half_edge_handle<Index> heh, const tf::point<Real, 3> &pt,
    MinQualityT min_quality = tf::none,
    EdgeLengthT max_edge_length_sq = tf::none,
    bool check_normals = false) -> bool {
  constexpr bool Quality = !std::is_same_v<MinQualityT, tf::none_t>;
  constexpr bool EdgeLength = !std::is_same_v<EdgeLengthT, tf::none_t>;
  auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
  auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();

  auto frame = tf::frame_of(points);

  Index link[128];
  int link_size = 0;

  Real max_old_e2 = Real(0);
  if constexpr (EdgeLength) {
    auto d = points[v0] - points[v1];
    max_old_e2 =
        std::max(Real(max_edge_length_sq),
                 tf::transformed(d, frame).length2());
  }
  Real max_new_e2 = 0;

  // A ring walk that hits an invalid handle (boundary or non-manifold fan)
  // cannot be inspected -- reject conservatively. The permissive "can't
  // check, allow" alternative lets collapses at exactly the irregular spots
  // (e.g. along a non-manifold imprint curve) go through ungated.
  // v0's ring
  {
    auto cur = he.rotated(heh);
    if (!cur.is_valid())
      return false;
    while (cur != heh) {
      if (link_size >= 128)
        return false;
      auto v = he.end_vertex_handle(tf::unsafe, cur).id();
      if constexpr (EdgeLength) {
        auto dold = points[v0] - points[v];
        auto dnew = points[v] - pt;
        max_old_e2 = std::max(max_old_e2,
                              tf::transformed(dold, frame).length2());
        max_new_e2 = std::max(max_new_e2,
                              tf::transformed(dnew, frame).length2());
      }
      link[link_size++] = v;
      cur = he.rotated(cur);
      if (!cur.is_valid())
        return false;
    }
  }

  int v0_link_end = link_size;

  // v1's ring (skip shared vertices)
  {
    auto skip = he.rotated(he.opposite(tf::unsafe, heh));
    if (!skip.is_valid())
      return false;
    auto cur = he.rotated(skip);
    if (!cur.is_valid())
      return false;
    auto opp = he.opposite(tf::unsafe, heh);
    while (cur != opp) {
      if (link_size >= 128)
        return false;
      auto v = he.end_vertex_handle(tf::unsafe, cur).id();
      if constexpr (EdgeLength) {
        auto dold = points[v1] - points[v];
        auto dnew = points[v] - pt;
        max_old_e2 = std::max(max_old_e2,
                              tf::transformed(dold, frame).length2());
        max_new_e2 = std::max(max_new_e2,
                              tf::transformed(dnew, frame).length2());
      }
      link[link_size++] = v;
      cur = he.rotated(cur);
      if (!cur.is_valid())
        return false;
    }
  }

  if constexpr (EdgeLength) {
    if (max_new_e2 > max_old_e2)
      return false;
  }

  // Dedup if closed ring
  if (link_size > 1 && link[link_size - 1] == link[0])
    --link_size;
  if (link_size < 3)
    return true;

  // The whole geometric loop runs in double regardless of the point type:
  // near-degenerate faces have sub-ulp float areas, so their float
  // orientation signs are noise (the fold/normal tests become coin flips).
  constexpr double max_area_ratio = 1e8;
  // Reject when adjacent new-face normals are within 10° of antiparallel
  // (i.e. surface dihedral < 10° — the near-fold zone). cos²(10°) ≈ 0.9698.
  constexpr double cos2_fold_tol = 0.9698463;

  // Triangle quality q = (2/sqrt3) * (2*area) / max_edge^2 in (0,1] (1 =
  // equilateral). Allow iff worst NEW q >= min(floor_q, worst OLD q):
  //   min_quality < 0 -> disabled; == 0 -> never worsen; > 0 -> quality floor.
  const double kq = tf::two_over_sqrt_3<double>;
  double min_old_q = std::numeric_limits<double>::max();
  double min_new_q = std::numeric_limits<double>::max();
  bool q_active = false;
  double floor_q = 0;
  if constexpr (Quality) {
    double mq = double(min_quality);
    q_active = mq >= 0;
    floor_q = (mq == 0) ? std::numeric_limits<double>::max() : mq;
  }

  auto dvec = [&](const auto &a, const auto &b) {
    return tf::vector<double, 3>{double(a[0]) - double(b[0]),
                                 double(a[1]) - double(b[1]),
                                 double(a[2]) - double(b[2])};
  };

  // Aggregate of the link's old-face normals: the substitute reference for
  // old faces too degenerate to vouch for themselves. Built lazily -- the
  // degenerate-old-face path is rare and the aggregate costs a full ring.
  tf::vector<double, 3> old_ref = tf::zero;
  bool old_ref_built = false;
  auto build_old_ref = [&]() {
    for (int i = 0; i < link_size; ++i) {
      int nxt = (i + 1) % link_size;
      bool i_in_v0 = i < v0_link_end;
      if (i_in_v0 != (nxt < v0_link_end))
        continue;
      auto center = i_in_v0 ? v0 : v1;
      old_ref += tf::cross(dvec(points[link[i]], points[center]),
                           dvec(points[link[nxt]], points[center]));
    }
    old_ref_built = true;
  };

  for (int i = 0; i < link_size; ++i) {
    int prv = (i + link_size - 1) % link_size;
    int nxt = (i + 1) % link_size;

    auto e01 = dvec(points[link[prv]], pt);
    auto e02 = dvec(points[link[i]], pt);
    auto n1 = tf::cross(e01, e02);

    auto e03 = dvec(points[link[nxt]], pt);
    auto n2 = tf::cross(e02, e03);

    double l1 = n1.length2();
    double l2 = n2.length2();
    if (std::max(l1, l2) >= max_area_ratio * std::min(l1, l2))
      return false;

    double d = tf::dot(n1, n2);
    if (d <= 0 && d * d > cos2_fold_tol * l1 * l2)
      return false;

    if (check_normals) {
      // Per-face old-vs-new normal (CGAL Bounded_normal_change). The new face is
      // (pt, link[i], link[nxt]) with normal n2; its OLD form is
      // (center, link[i], link[nxt]) where center is the original ring owner --
      // a real old face only when both link vertices come from the same ring.
      // A new face below the determinable floor has a noise sign: reject. A
      // degenerate old face's normal is noise too: the aggregate substitutes.
      if (l2 <= 1e-24 * tf::dot(e02, e02) * tf::dot(e03, e03))
        return false;
      bool i_in_v0 = i < v0_link_end;
      bool nxt_in_v0 = nxt < v0_link_end;
      if (i_in_v0 == nxt_in_v0) {
        auto center = i_in_v0 ? v0 : v1;
        auto oa = dvec(points[link[i]], points[center]);
        auto ob = dvec(points[link[nxt]], points[center]);
        auto old_n = tf::cross(oa, ob);
        bool healthy =
            tf::dot(old_n, old_n) > 1e-14 * tf::dot(oa, oa) * tf::dot(ob, ob);
        if (!healthy && !old_ref_built)
          build_old_ref();
        if (tf::dot(healthy ? old_n : old_ref, n2) <= 0)
          return false; // face normal inverts
      }
    }

    if constexpr (Quality) {
      if (!q_active)
        continue;
      // New face (pt, link[i], link[nxt]) quality (lower = worse).
      {
        auto fc = dvec(points[link[nxt]], points[link[i]]);
        double new_me2 =
            std::max({tf::dot(e02, e02), tf::dot(e03, e03), tf::dot(fc, fc)});
        if (new_me2 > 0) {
          double qn = kq * tf::sqrt(std::max(l2, 0.0)) / new_me2;
          min_new_q = std::min(min_new_q, qn);
        }
      }
      // Old face quality — only when both vertices in same ring.
      bool i_in_v0 = i < v0_link_end;
      bool nxt_in_v0 = nxt < v0_link_end;
      if (i_in_v0 == nxt_in_v0) {
        auto center = i_in_v0 ? v0 : v1;
        auto oa = dvec(points[link[i]], points[center]);
        auto ob = dvec(points[link[nxt]], points[center]);
        auto old_n = tf::cross(oa, ob);
        double old_a2 = old_n.length2();
        auto oc = dvec(points[link[nxt]], points[link[i]]);
        double old_me2 =
            std::max({tf::dot(oa, oa), tf::dot(ob, ob), tf::dot(oc, oc)});
        if (old_me2 > 0) {
          double qo = kq * tf::sqrt(std::max(old_a2, 0.0)) / old_me2;
          min_old_q = std::min(min_old_q, qo);
        }
      }
    }
  }

  if constexpr (Quality) {
    // Allow iff the worst new quality is no lower than min(floor_q, worst old).
    // Improvement always passes; floor_q = +inf (min_quality == 0) -> never
    // worsen; otherwise a quality floor that still permits matching an
    // already-worse ring.
    if (q_active && min_new_q < std::min(floor_q, min_old_q))
      return false;
  }

  return true;
}

/// @ingroup remesh
/// @brief Strict per-face normal-flip guard for an edge collapse.
///
/// Rejects the collapse if ANY surviving face incident to either endpoint
/// inverts -- its post-collapse normal opposes its pre-collapse normal
/// (dot <= 0). This is stricter than the two guards above and catches a case
/// neither does: a COHERENT inversion of a whole 1-ring fan, where every face
/// flips consistently. is_collapse_allowed's NormalFlip only rejects flipped
/// faces that oppose the AVERAGE new normal (a coherent flip agrees with its
/// own average), and is_collapse_allowed_dihedral only checks ADJACENT new
/// faces against each other (a coherent flip keeps them mutually consistent).
/// Use this when surface orientation must be preserved exactly (e.g. without
/// back-projection, where a collapse can fold a fan onto the wrong side).
///
/// Mirrors the 1-ring traversal of is_collapse_allowed; the two faces sharing
/// the collapsed edge are skipped (they are removed).
template <typename Index, typename PointsPolicy, typename Real>
auto is_collapse_normal_preserving(const tf::half_edges<Index> &he,
                                   const tf::points<PointsPolicy> &points,
                                   tf::half_edge_handle<Index> heh,
                                   const tf::point<Real, 3> &pt) -> bool {
  auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
  auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();

  // Faces (center, a, b) in `center`'s 1-ring; `other` is the far endpoint,
  // whose two shared faces (b == other) are the removed ones and are skipped.
  auto ring_ok = [&](tf::half_edge_handle<Index> outgoing, Index center,
                     Index other) -> bool {
    auto cur = he.rotated(outgoing);
    if (!cur.is_valid())
      return true; // open fan -> boundary handled elsewhere
    while (cur != outgoing) {
      auto a = he.end_vertex_handle(tf::unsafe, cur).id();
      auto b = he.end_vertex_handle(tf::unsafe, he.next(tf::unsafe, cur)).id();
      if (b != other) {
        auto ea = points[a] - points[center];
        auto eb = points[b] - points[center];
        auto fa = points[a] - pt;
        auto fb = points[b] - pt;
        if (tf::dot(tf::cross(ea, eb), tf::cross(fa, fb)) <= Real(0))
          return false; // this face inverts
      }
      cur = he.rotated(cur);
      if (!cur.is_valid())
        return true;
    }
    return true;
  };

  return ring_ok(heh, v0, v1) &&
         ring_ok(he.opposite(tf::unsafe, heh), v1, v0);
}

} // namespace tf::remesh
