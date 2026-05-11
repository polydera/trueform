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
/// @tparam AspectRatio Reject if worst aspect ratio increases.
/// @tparam EdgeLength  Reject if longest ring edge increases (or exceeds
///                     max_edge_length_sq when > 0).
template <bool NormalFlip = true, typename Index, typename PointsPolicy,
          typename Real, typename AspectRatioT = tf::none_t,
          typename EdgeLengthT = tf::none_t>
auto is_collapse_allowed(const tf::half_edges<Index> &he,
                         const tf::points<PointsPolicy> &points,
                         tf::half_edge_handle<Index> heh,
                         const tf::point<Real, 3> &pt,
                         AspectRatioT max_aspect_ratio = tf::none,
                         EdgeLengthT max_edge_length_sq = tf::none) -> bool {
  constexpr bool AspectRatio = !std::is_same_v<AspectRatioT, tf::none_t>;
  constexpr bool EdgeLength = !std::is_same_v<EdgeLengthT, tf::none_t>;
  auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
  auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();

  auto frame = tf::frame_of(points);

  Real max_old_ar = Real(0);
  bool ar_active = false;
  if constexpr (!std::is_same_v<AspectRatioT, tf::none_t>) {
    max_old_ar = Real(max_aspect_ratio);
    ar_active = max_old_ar >= Real(0);
  }
  Real max_new_ar = 0;
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

        if constexpr (NormalFlip || AspectRatio) {
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

          if constexpr (AspectRatio) {
            if (ar_active) {
              auto ec = points[b] - points[a];
              Real ec2 = tf::dot(ec, ec);

              Real old_a2 = old_n.length2();
              if (old_a2 > 0) {
                Real old_me2 =
                    std::max({tf::dot(ea, ea), tf::dot(eb, eb), ec2});
                max_old_ar = std::max(max_old_ar, old_me2 / old_a2);
              }

              Real new_a2 = new_n.length2();
              if (new_a2 > 0) {
                Real new_me2 =
                    std::max({tf::dot(fa, fa), tf::dot(fb, fb), ec2});
                max_new_ar = std::max(max_new_ar, new_me2 / new_a2);
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

        if constexpr (NormalFlip || AspectRatio) {
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

          if constexpr (AspectRatio) {
            if (ar_active) {
              auto ec = points[b] - points[a];
              Real ec2 = tf::dot(ec, ec);

              Real old_a2 = old_n.length2();
              if (old_a2 > 0) {
                Real old_me2 =
                    std::max({tf::dot(ea, ea), tf::dot(eb, eb), ec2});
                max_old_ar = std::max(max_old_ar, old_me2 / old_a2);
              }

              Real new_a2 = new_n.length2();
              if (new_a2 > 0) {
                Real new_me2 =
                    std::max({tf::dot(fa, fa), tf::dot(fb, fb), ec2});
                max_new_ar = std::max(max_new_ar, new_me2 / new_a2);
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

  if constexpr (AspectRatio) {
    if (ar_active && max_new_ar > max_old_ar)
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
/// @tparam AspectRatioT Pass a Real value to enable the aspect ratio
///                      check (used as floor for max old AR).
///                      Default tf::none disables the check.
/// @tparam EdgeLengthT  Pass a Real value to enable the edge length
///                      check (used as floor for max old edge²).
///                      Default tf::none disables the check.
template <typename Index, typename PointsPolicy, typename Real,
          typename AspectRatioT = tf::none_t,
          typename EdgeLengthT = tf::none_t>
auto is_collapse_allowed_dihedral(
    const tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    tf::half_edge_handle<Index> heh, const tf::point<Real, 3> &pt,
    AspectRatioT max_aspect_ratio = tf::none,
    EdgeLengthT max_edge_length_sq = tf::none) -> bool {
  constexpr bool AspectRatio = !std::is_same_v<AspectRatioT, tf::none_t>;
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

  // v0's ring
  {
    auto cur = he.rotated(heh);
    if (!cur.is_valid())
      return true;
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
        return true;
    }
  }

  int v0_link_end = link_size;

  // v1's ring (skip shared vertices)
  {
    auto skip = he.rotated(he.opposite(tf::unsafe, heh));
    if (!skip.is_valid())
      return true;
    auto cur = he.rotated(skip);
    if (!cur.is_valid())
      return true;
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
        return true;
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

  constexpr Real max_area_ratio = Real(1e8);
  // Reject when adjacent new-face normals are within 10° of antiparallel
  // (i.e. surface dihedral < 10° — the near-fold zone). cos²(10°) ≈ 0.9698.
  constexpr Real cos2_fold_tol = Real(0.9698463);

  Real max_old_ar = Real(0);
  bool ar_active = false;
  if constexpr (AspectRatio) {
    max_old_ar = Real(max_aspect_ratio);
    ar_active = max_old_ar >= Real(0);
  }
  Real max_new_ar = 0;

  for (int i = 0; i < link_size; ++i) {
    int prv = (i + link_size - 1) % link_size;
    int nxt = (i + 1) % link_size;

    auto e01 = points[link[prv]] - pt;
    auto e02 = points[link[i]] - pt;
    auto n1 = tf::cross(e01, e02);

    auto e03 = points[link[nxt]] - pt;
    auto n2 = tf::cross(e02, e03);

    Real l1 = n1.length2();
    Real l2 = n2.length2();
    if (std::max(l1, l2) >= max_area_ratio * std::min(l1, l2))
      return false;

    Real d = tf::dot(n1, n2);
    if (d <= Real(0) && d * d > cos2_fold_tol * l1 * l2)
      return false;

    if constexpr (AspectRatio) {
      if (!ar_active)
        continue;
      // New face (pt, link[i], link[nxt])
      if (l2 > 0) {
        auto fc = points[link[nxt]] - points[link[i]];
        Real new_me2 =
            std::max({tf::dot(e02, e02), tf::dot(e03, e03), tf::dot(fc, fc)});
        max_new_ar = std::max(max_new_ar, new_me2 / l2);
      }
      // Old face — only when both vertices in same ring
      bool i_in_v0 = i < v0_link_end;
      bool nxt_in_v0 = nxt < v0_link_end;
      if (i_in_v0 == nxt_in_v0) {
        auto center = i_in_v0 ? v0 : v1;
        auto oa = points[link[i]] - points[center];
        auto ob = points[link[nxt]] - points[center];
        auto old_n = tf::cross(oa, ob);
        Real old_a2 = old_n.length2();
        if (old_a2 > 0) {
          auto oc = points[link[nxt]] - points[link[i]];
          Real old_me2 =
              std::max({tf::dot(oa, oa), tf::dot(ob, ob), tf::dot(oc, oc)});
          max_old_ar = std::max(max_old_ar, old_me2 / old_a2);
        }
      }
    }
  }

  if constexpr (AspectRatio) {
    if (ar_active && max_new_ar > max_old_ar)
      return false;
  }

  return true;
}

} // namespace tf::remesh
