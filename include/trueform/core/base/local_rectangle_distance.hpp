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
#include "../epsilon.hpp"
#include "../sqrt.hpp"
#include <algorithm>
#include <array>

namespace tf::core::impl {

template <typename T>
auto segment_closest_params(T &t, T &u, T a, T b, T a_dot_b, T a_dot_t,
                            T b_dot_t) -> void {
  T denom = T(1) - a_dot_b * a_dot_b;

  if (denom == T(0))
    t = T(0);
  else {
    t = (a_dot_t - b_dot_t * a_dot_b) / denom;
    t = std::clamp(t, T(0), a);
  }

  u = t * a_dot_b - b_dot_t;
  if (u < T(0)) {
    u = T(0);
    t = std::clamp(a_dot_t, T(0), a);
  } else if (u > b) {
    u = b;
    t = std::clamp(u * a_dot_b + a_dot_t, T(0), a);
  }
}

template <typename T>
auto is_in_voronoi_region(T a, T b, T anorm_dot_b, T anorm_dot_t, T a_dot_b,
                          T a_dot_t, T b_dot_t) -> bool {
  using std::fabs;
  if (fabs(anorm_dot_b) < tf::epsilon<T>)
    return false;

  T t, u, v;

  u = std::clamp(-anorm_dot_t / anorm_dot_b, T(0), b);
  t = std::clamp(u * a_dot_b + a_dot_t, T(0), a);

  v = t * a_dot_b - b_dot_t;

  if (anorm_dot_b > T(0)) {
    if (v > (u + tf::epsilon<T>))
      return true;
  } else {
    if (v < (u - tf::epsilon<T>))
      return true;
  }
  return false;
}

} // namespace tf::core::impl

namespace tf::core {

template <typename T>
auto local_rectangle_distance2(const std::array<std::array<T, 3>, 3> &r_ab,
                               const std::array<T, 3> &t_ab,
                               const std::array<T, 2> a,
                               const std::array<T, 2> b) -> T {
  using std::max;
  using std::min;

  T a0_dot_b0 = r_ab[0][0];
  T a0_dot_b1 = r_ab[0][1];
  T a1_dot_b0 = r_ab[1][0];
  T a1_dot_b1 = r_ab[1][1];

  T a_a0_dot_b0 = a[0] * a0_dot_b0;
  T a_a0_dot_b1 = a[0] * a0_dot_b1;
  T a_a1_dot_b0 = a[1] * a1_dot_b0;
  T a_a1_dot_b1 = a[1] * a1_dot_b1;
  T b_a0_dot_b0 = b[0] * a0_dot_b0;
  T b_a1_dot_b0 = b[0] * a1_dot_b0;
  T b_a0_dot_b1 = b[1] * a0_dot_b1;
  T b_a1_dot_b1 = b[1] * a1_dot_b1;

  std::array<T, 3> t_ba;
  t_ba[0] = r_ab[0][0] * t_ab[0] + r_ab[1][0] * t_ab[1] + r_ab[2][0] * t_ab[2];
  t_ba[1] = r_ab[0][1] * t_ab[0] + r_ab[1][1] * t_ab[1] + r_ab[2][1] * t_ab[2];
  t_ba[2] = r_ab[0][2] * t_ab[0] + r_ab[1][2] * t_ab[1] + r_ab[2][2] * t_ab[2];

  std::array<T, 3> d;
  T t, u;

  T all_x = -t_ba[0];
  T alu_x = all_x + a_a1_dot_b0;
  T aul_x = all_x + a_a0_dot_b0;
  T auu_x = alu_x + a_a0_dot_b0;

  T la1_lx, la1_ux, ua1_lx, ua1_ux;
  if (all_x < alu_x) {
    la1_lx = all_x;
    la1_ux = alu_x;
    ua1_lx = aul_x;
    ua1_ux = auu_x;
  } else {
    la1_lx = alu_x;
    la1_ux = all_x;
    ua1_lx = auu_x;
    ua1_ux = aul_x;
  }

  T bll_x = t_ab[0];
  T blu_x = bll_x + b_a0_dot_b1;
  T bul_x = bll_x + b_a0_dot_b0;
  T buu_x = blu_x + b_a0_dot_b0;

  T lb1_lx, lb1_ux, ub1_lx, ub1_ux;
  if (bll_x < blu_x) {
    lb1_lx = bll_x;
    lb1_ux = blu_x;
    ub1_lx = bul_x;
    ub1_ux = buu_x;
  } else {
    lb1_lx = blu_x;
    lb1_ux = bll_x;
    ub1_lx = buu_x;
    ub1_ux = bul_x;
  }

  if ((ua1_ux > b[0]) && (ub1_ux > a[0])) {
    if (((ua1_lx > b[0]) ||
         impl::is_in_voronoi_region(
             b[1], a[1], a1_dot_b0, a_a0_dot_b0 - b[0] - t_ba[0], a1_dot_b1,
             a_a0_dot_b1 - t_ba[1], -t_ab[1] - b_a1_dot_b0)) &&
        ((ub1_lx > a[0]) ||
         impl::is_in_voronoi_region(
             a[1], b[1], a0_dot_b1, t_ab[0] + b_a0_dot_b0 - a[0], a1_dot_b1,
             t_ab[1] + b_a1_dot_b0, t_ba[1] - a_a0_dot_b1))) {
      impl::segment_closest_params(t, u, a[1], b[1], a1_dot_b1,
                                   t_ab[1] + b_a1_dot_b0,
                                   t_ba[1] - a_a0_dot_b1);
      d[0] = t_ab[0] + r_ab[0][0] * b[0] + r_ab[0][1] * u - a[0];
      d[1] = t_ab[1] + r_ab[1][0] * b[0] + r_ab[1][1] * u - t;
      d[2] = t_ab[2] + r_ab[2][0] * b[0] + r_ab[2][1] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((ua1_lx < T(0)) && (lb1_ux > a[0])) {
    if (((ua1_ux < T(0)) || impl::is_in_voronoi_region(
                                b[1], a[1], -a1_dot_b0, t_ba[0] - a_a0_dot_b0,
                                a1_dot_b1, a_a0_dot_b1 - t_ba[1], -t_ab[1])) &&
        ((lb1_lx > a[0]) || impl::is_in_voronoi_region(
                                a[1], b[1], a0_dot_b1, t_ab[0] - a[0],
                                a1_dot_b1, t_ab[1], t_ba[1] - a_a0_dot_b1))) {
      impl::segment_closest_params(t, u, a[1], b[1], a1_dot_b1, t_ab[1],
                                   t_ba[1] - a_a0_dot_b1);
      d[0] = t_ab[0] + r_ab[0][1] * u - a[0];
      d[1] = t_ab[1] + r_ab[1][1] * u - t;
      d[2] = t_ab[2] + r_ab[2][1] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((la1_ux > b[0]) && (ub1_lx < T(0))) {
    if (((la1_lx > b[0]) || impl::is_in_voronoi_region(
                                b[1], a[1], a1_dot_b0, -t_ba[0] - b[0],
                                a1_dot_b1, -t_ba[1], -t_ab[1] - b_a1_dot_b0)) &&
        ((ub1_ux < T(0)) || impl::is_in_voronoi_region(
                                a[1], b[1], -a0_dot_b1, -t_ab[0] - b_a0_dot_b0,
                                a1_dot_b1, t_ab[1] + b_a1_dot_b0, t_ba[1]))) {
      impl::segment_closest_params(t, u, a[1], b[1], a1_dot_b1,
                                   t_ab[1] + b_a1_dot_b0, t_ba[1]);
      d[0] = t_ab[0] + r_ab[0][0] * b[0] + r_ab[0][1] * u;
      d[1] = t_ab[1] + r_ab[1][0] * b[0] + r_ab[1][1] * u - t;
      d[2] = t_ab[2] + r_ab[2][0] * b[0] + r_ab[2][1] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((la1_lx < T(0)) && (lb1_lx < T(0))) {
    if (((la1_ux < T(0)) ||
         impl::is_in_voronoi_region(b[1], a[1], -a1_dot_b0, t_ba[0], a1_dot_b1,
                                    -t_ba[1], -t_ab[1])) &&
        ((lb1_ux < T(0)) ||
         impl::is_in_voronoi_region(a[1], b[1], -a0_dot_b1, -t_ab[0], a1_dot_b1,
                                    t_ab[1], t_ba[1]))) {
      impl::segment_closest_params(t, u, a[1], b[1], a1_dot_b1, t_ab[1],
                                   t_ba[1]);
      d[0] = t_ab[0] + r_ab[0][1] * u;
      d[1] = t_ab[1] + r_ab[1][1] * u - t;
      d[2] = t_ab[2] + r_ab[2][1] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  T all_y = -t_ba[1];
  T alu_y = all_y + a_a1_dot_b1;
  T aul_y = all_y + a_a0_dot_b1;
  T auu_y = alu_y + a_a0_dot_b1;

  T la1_ly, la1_uy, ua1_ly, ua1_uy;
  if (all_y < alu_y) {
    la1_ly = all_y;
    la1_uy = alu_y;
    ua1_ly = aul_y;
    ua1_uy = auu_y;
  } else {
    la1_ly = alu_y;
    la1_uy = all_y;
    ua1_ly = auu_y;
    ua1_uy = aul_y;
  }

  T lb0_lx, lb0_ux, ub0_lx, ub0_ux;
  if (bll_x < bul_x) {
    lb0_lx = bll_x;
    lb0_ux = bul_x;
    ub0_lx = blu_x;
    ub0_ux = buu_x;
  } else {
    lb0_lx = bul_x;
    lb0_ux = bll_x;
    ub0_lx = buu_x;
    ub0_ux = blu_x;
  }

  if ((ua1_uy > b[1]) && (ub0_ux > a[0])) {
    if (((ua1_ly > b[1]) ||
         impl::is_in_voronoi_region(
             b[0], a[1], a1_dot_b1, a_a0_dot_b1 - t_ba[1] - b[1], a1_dot_b0,
             a_a0_dot_b0 - t_ba[0], -t_ab[1] - b_a1_dot_b1)) &&
        ((ub0_lx > a[0]) ||
         impl::is_in_voronoi_region(
             a[1], b[0], a0_dot_b0, t_ab[0] - a[0] + b_a0_dot_b1, a1_dot_b0,
             t_ab[1] + b_a1_dot_b1, t_ba[0] - a_a0_dot_b0))) {
      impl::segment_closest_params(t, u, a[1], b[0], a1_dot_b0,
                                   t_ab[1] + b_a1_dot_b1,
                                   t_ba[0] - a_a0_dot_b0);
      d[0] = t_ab[0] + r_ab[0][1] * b[1] + r_ab[0][0] * u - a[0];
      d[1] = t_ab[1] + r_ab[1][1] * b[1] + r_ab[1][0] * u - t;
      d[2] = t_ab[2] + r_ab[2][1] * b[1] + r_ab[2][0] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((ua1_ly < T(0)) && (lb0_ux > a[0])) {
    if (((ua1_uy < T(0)) || impl::is_in_voronoi_region(
                                b[0], a[1], -a1_dot_b1, t_ba[1] - a_a0_dot_b1,
                                a1_dot_b0, a_a0_dot_b0 - t_ba[0], -t_ab[1])) &&
        ((lb0_lx > a[0]) || impl::is_in_voronoi_region(
                                a[1], b[0], a0_dot_b0, t_ab[0] - a[0],
                                a1_dot_b0, t_ab[1], t_ba[0] - a_a0_dot_b0))) {
      impl::segment_closest_params(t, u, a[1], b[0], a1_dot_b0, t_ab[1],
                                   t_ba[0] - a_a0_dot_b0);
      d[0] = t_ab[0] + r_ab[0][0] * u - a[0];
      d[1] = t_ab[1] + r_ab[1][0] * u - t;
      d[2] = t_ab[2] + r_ab[2][0] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((la1_uy > b[1]) && (ub0_lx < T(0))) {
    if (((la1_ly > b[1]) || impl::is_in_voronoi_region(
                                b[0], a[1], a1_dot_b1, -t_ba[1] - b[1],
                                a1_dot_b0, -t_ba[0], -t_ab[1] - b_a1_dot_b1)) &&
        ((ub0_ux < T(0)) || impl::is_in_voronoi_region(
                                a[1], b[0], -a0_dot_b0, -t_ab[0] - b_a0_dot_b1,
                                a1_dot_b0, t_ab[1] + b_a1_dot_b1, t_ba[0]))) {
      impl::segment_closest_params(t, u, a[1], b[0], a1_dot_b0,
                                   t_ab[1] + b_a1_dot_b1, t_ba[0]);
      d[0] = t_ab[0] + r_ab[0][1] * b[1] + r_ab[0][0] * u;
      d[1] = t_ab[1] + r_ab[1][1] * b[1] + r_ab[1][0] * u - t;
      d[2] = t_ab[2] + r_ab[2][1] * b[1] + r_ab[2][0] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((la1_ly < T(0)) && (lb0_lx < T(0))) {
    if (((la1_uy < T(0)) ||
         impl::is_in_voronoi_region(b[0], a[1], -a1_dot_b1, t_ba[1], a1_dot_b0,
                                    -t_ba[0], -t_ab[1])) &&
        ((lb0_ux < T(0)) ||
         impl::is_in_voronoi_region(a[1], b[0], -a0_dot_b0, -t_ab[0], a1_dot_b0,
                                    t_ab[1], t_ba[0]))) {
      impl::segment_closest_params(t, u, a[1], b[0], a1_dot_b0, t_ab[1],
                                   t_ba[0]);
      d[0] = t_ab[0] + r_ab[0][0] * u;
      d[1] = t_ab[1] + r_ab[1][0] * u - t;
      d[2] = t_ab[2] + r_ab[2][0] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  T bll_y = t_ab[1];
  T blu_y = bll_y + b_a1_dot_b1;
  T bul_y = bll_y + b_a1_dot_b0;
  T buu_y = blu_y + b_a1_dot_b0;

  T la0_lx, la0_ux, ua0_lx, ua0_ux;
  if (all_x < aul_x) {
    la0_lx = all_x;
    la0_ux = aul_x;
    ua0_lx = alu_x;
    ua0_ux = auu_x;
  } else {
    la0_lx = aul_x;
    la0_ux = all_x;
    ua0_lx = auu_x;
    ua0_ux = alu_x;
  }

  T lb1_ly, lb1_uy, ub1_ly, ub1_uy;
  if (bll_y < blu_y) {
    lb1_ly = bll_y;
    lb1_uy = blu_y;
    ub1_ly = bul_y;
    ub1_uy = buu_y;
  } else {
    lb1_ly = blu_y;
    lb1_uy = bll_y;
    ub1_ly = buu_y;
    ub1_uy = bul_y;
  }

  if ((ua0_ux > b[0]) && (ub1_uy > a[1])) {
    if (((ua0_lx > b[0]) ||
         impl::is_in_voronoi_region(
             b[1], a[0], a0_dot_b0, a_a1_dot_b0 - t_ba[0] - b[0], a0_dot_b1,
             a_a1_dot_b1 - t_ba[1], -t_ab[0] - b_a0_dot_b0)) &&
        ((ub1_ly > a[1]) ||
         impl::is_in_voronoi_region(
             a[0], b[1], a1_dot_b1, t_ab[1] - a[1] + b_a1_dot_b0, a0_dot_b1,
             t_ab[0] + b_a0_dot_b0, t_ba[1] - a_a1_dot_b1))) {
      impl::segment_closest_params(t, u, a[0], b[1], a0_dot_b1,
                                   t_ab[0] + b_a0_dot_b0,
                                   t_ba[1] - a_a1_dot_b1);
      d[0] = t_ab[0] + r_ab[0][0] * b[0] + r_ab[0][1] * u - t;
      d[1] = t_ab[1] + r_ab[1][0] * b[0] + r_ab[1][1] * u - a[1];
      d[2] = t_ab[2] + r_ab[2][0] * b[0] + r_ab[2][1] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((ua0_lx < T(0)) && (lb1_uy > a[1])) {
    if (((ua0_ux < T(0)) || impl::is_in_voronoi_region(
                                b[1], a[0], -a0_dot_b0, t_ba[0] - a_a1_dot_b0,
                                a0_dot_b1, a_a1_dot_b1 - t_ba[1], -t_ab[0])) &&
        ((lb1_ly > a[1]) || impl::is_in_voronoi_region(
                                a[0], b[1], a1_dot_b1, t_ab[1] - a[1],
                                a0_dot_b1, t_ab[0], t_ba[1] - a_a1_dot_b1))) {
      impl::segment_closest_params(t, u, a[0], b[1], a0_dot_b1, t_ab[0],
                                   t_ba[1] - a_a1_dot_b1);
      d[0] = t_ab[0] + r_ab[0][1] * u - t;
      d[1] = t_ab[1] + r_ab[1][1] * u - a[1];
      d[2] = t_ab[2] + r_ab[2][1] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((la0_ux > b[0]) && (ub1_ly < T(0))) {
    if (((la0_lx > b[0]) || impl::is_in_voronoi_region(
                                b[1], a[0], a0_dot_b0, -b[0] - t_ba[0],
                                a0_dot_b1, -t_ba[1], -b_a0_dot_b0 - t_ab[0])) &&
        ((ub1_uy < T(0)) || impl::is_in_voronoi_region(
                                a[0], b[1], -a1_dot_b1, -t_ab[1] - b_a1_dot_b0,
                                a0_dot_b1, t_ab[0] + b_a0_dot_b0, t_ba[1]))) {
      impl::segment_closest_params(t, u, a[0], b[1], a0_dot_b1,
                                   t_ab[0] + b_a0_dot_b0, t_ba[1]);
      d[0] = t_ab[0] + r_ab[0][0] * b[0] + r_ab[0][1] * u - t;
      d[1] = t_ab[1] + r_ab[1][0] * b[0] + r_ab[1][1] * u;
      d[2] = t_ab[2] + r_ab[2][0] * b[0] + r_ab[2][1] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((la0_lx < T(0)) && (lb1_ly < T(0))) {
    if (((la0_ux < T(0)) ||
         impl::is_in_voronoi_region(b[1], a[0], -a0_dot_b0, t_ba[0], a0_dot_b1,
                                    -t_ba[1], -t_ab[0])) &&
        ((lb1_uy < T(0)) ||
         impl::is_in_voronoi_region(a[0], b[1], -a1_dot_b1, -t_ab[1], a0_dot_b1,
                                    t_ab[0], t_ba[1]))) {
      impl::segment_closest_params(t, u, a[0], b[1], a0_dot_b1, t_ab[0],
                                   t_ba[1]);
      d[0] = t_ab[0] + r_ab[0][1] * u - t;
      d[1] = t_ab[1] + r_ab[1][1] * u;
      d[2] = t_ab[2] + r_ab[2][1] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  T la0_ly, la0_uy, ua0_ly, ua0_uy;
  if (all_y < aul_y) {
    la0_ly = all_y;
    la0_uy = aul_y;
    ua0_ly = alu_y;
    ua0_uy = auu_y;
  } else {
    la0_ly = aul_y;
    la0_uy = all_y;
    ua0_ly = auu_y;
    ua0_uy = alu_y;
  }

  T lb0_ly, lb0_uy, ub0_ly, ub0_uy;
  if (bll_y < bul_y) {
    lb0_ly = bll_y;
    lb0_uy = bul_y;
    ub0_ly = blu_y;
    ub0_uy = buu_y;
  } else {
    lb0_ly = bul_y;
    lb0_uy = bll_y;
    ub0_ly = buu_y;
    ub0_uy = blu_y;
  }

  if ((ua0_uy > b[1]) && (ub0_uy > a[1])) {
    if (((ua0_ly > b[1]) ||
         impl::is_in_voronoi_region(
             b[0], a[0], a0_dot_b1, a_a1_dot_b1 - t_ba[1] - b[1], a0_dot_b0,
             a_a1_dot_b0 - t_ba[0], -t_ab[0] - b_a0_dot_b1)) &&
        ((ub0_ly > a[1]) ||
         impl::is_in_voronoi_region(
             a[0], b[0], a1_dot_b0, t_ab[1] - a[1] + b_a1_dot_b1, a0_dot_b0,
             t_ab[0] + b_a0_dot_b1, t_ba[0] - a_a1_dot_b0))) {
      impl::segment_closest_params(t, u, a[0], b[0], a0_dot_b0,
                                   t_ab[0] + b_a0_dot_b1,
                                   t_ba[0] - a_a1_dot_b0);
      d[0] = t_ab[0] + r_ab[0][1] * b[1] + r_ab[0][0] * u - t;
      d[1] = t_ab[1] + r_ab[1][1] * b[1] + r_ab[1][0] * u - a[1];
      d[2] = t_ab[2] + r_ab[2][1] * b[1] + r_ab[2][0] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((ua0_ly < T(0)) && (lb0_uy > a[1])) {
    if (((ua0_uy < T(0)) || impl::is_in_voronoi_region(
                                b[0], a[0], -a0_dot_b1, t_ba[1] - a_a1_dot_b1,
                                a0_dot_b0, a_a1_dot_b0 - t_ba[0], -t_ab[0])) &&
        ((lb0_ly > a[1]) || impl::is_in_voronoi_region(
                                a[0], b[0], a1_dot_b0, t_ab[1] - a[1],
                                a0_dot_b0, t_ab[0], t_ba[0] - a_a1_dot_b0))) {
      impl::segment_closest_params(t, u, a[0], b[0], a0_dot_b0, t_ab[0],
                                   t_ba[0] - a_a1_dot_b0);
      d[0] = t_ab[0] + r_ab[0][0] * u - t;
      d[1] = t_ab[1] + r_ab[1][0] * u - a[1];
      d[2] = t_ab[2] + r_ab[2][0] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((la0_uy > b[1]) && (ub0_ly < T(0))) {
    if (((la0_ly > b[1]) || impl::is_in_voronoi_region(
                                b[0], a[0], a0_dot_b1, -t_ba[1] - b[1],
                                a0_dot_b0, -t_ba[0], -t_ab[0] - b_a0_dot_b1)) &&
        ((ub0_uy < T(0)) || impl::is_in_voronoi_region(
                                a[0], b[0], -a1_dot_b0, -t_ab[1] - b_a1_dot_b1,
                                a0_dot_b0, t_ab[0] + b_a0_dot_b1, t_ba[0]))) {
      impl::segment_closest_params(t, u, a[0], b[0], a0_dot_b0,
                                   t_ab[0] + b_a0_dot_b1, t_ba[0]);
      d[0] = t_ab[0] + r_ab[0][1] * b[1] + r_ab[0][0] * u - t;
      d[1] = t_ab[1] + r_ab[1][1] * b[1] + r_ab[1][0] * u;
      d[2] = t_ab[2] + r_ab[2][1] * b[1] + r_ab[2][0] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  if ((la0_ly < T(0)) && (lb0_ly < T(0))) {
    if (((la0_uy < T(0)) ||
         impl::is_in_voronoi_region(b[0], a[0], -a0_dot_b1, t_ba[1], a0_dot_b0,
                                    -t_ba[0], -t_ab[0])) &&
        ((lb0_uy < T(0)) ||
         impl::is_in_voronoi_region(a[0], b[0], -a1_dot_b0, -t_ab[1], a0_dot_b0,
                                    t_ab[0], t_ba[0]))) {
      impl::segment_closest_params(t, u, a[0], b[0], a0_dot_b0, t_ab[0],
                                   t_ba[0]);
      d[0] = t_ab[0] + r_ab[0][0] * u - t;
      d[1] = t_ab[1] + r_ab[1][0] * u;
      d[2] = t_ab[2] + r_ab[2][0] * u;
      return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
  }

  T sep1, sep2;

  if (t_ab[2] > T(0)) {
    sep1 = t_ab[2];
    if (r_ab[2][0] < T(0))
      sep1 += b[0] * r_ab[2][0];
    if (r_ab[2][1] < T(0))
      sep1 += b[1] * r_ab[2][1];
  } else {
    sep1 = -t_ab[2];
    if (r_ab[2][0] > T(0))
      sep1 -= b[0] * r_ab[2][0];
    if (r_ab[2][1] > T(0))
      sep1 -= b[1] * r_ab[2][1];
  }

  if (t_ba[2] < T(0)) {
    sep2 = -t_ba[2];
    if (r_ab[0][2] < T(0))
      sep2 += a[0] * r_ab[0][2];
    if (r_ab[1][2] < T(0))
      sep2 += a[1] * r_ab[1][2];
  } else {
    sep2 = t_ba[2];
    if (r_ab[0][2] > T(0))
      sep2 -= a[0] * r_ab[0][2];
    if (r_ab[1][2] > T(0))
      sep2 -= a[1] * r_ab[1][2];
  }

  T sep = (sep1 > sep2 ? sep1 : sep2);
  if (sep < T(0))
    sep = T(0);
  return sep * sep;
}

template <typename T>
auto local_rectangle_distance(const std::array<std::array<T, 3>, 3> &r_ab,
                              const std::array<T, 3> &t_ab,
                              const std::array<T, 2> a,
                              const std::array<T, 2> b) -> T {
  return tf::sqrt(local_rectangle_distance2(r_ab, t_ab, a, b));
}

/// @brief Squared distance from point in local coordinates to axis-aligned box
/// [0, extent[0]] x [0, extent[1]] x ... x [0, extent[N-1]]
template <typename T, std::size_t N>
auto local_point_box_distance2(const std::array<T, N> &local_pt,
                               const std::array<T, N> &extent) -> T {
  T dist2 = T(0);
  for (std::size_t i = 0; i < N; ++i) {
    T d =
        std::max(-local_pt[i], T(0)) + std::max(local_pt[i] - extent[i], T(0));
    dist2 += d * d;
  }
  return dist2;
}

/// @brief Distance from point in local coordinates to axis-aligned box
template <typename T, std::size_t N>
auto local_point_box_distance(const std::array<T, N> &local_pt,
                              const std::array<T, N> &extent) -> T {
  return tf::sqrt(local_point_box_distance2(local_pt, extent));
}

/// @brief Squared distance from point in local coordinates to rectangle
/// Rectangle [0, length[0]] x [0, length[1]] at z=0
template <typename T>
auto local_point_rectangle_distance2(const std::array<T, 3> &local_pt,
                                     const std::array<T, 2> &length) -> T {
  T dx = std::max(-local_pt[0], T(0)) + std::max(local_pt[0] - length[0], T(0));
  T dy = std::max(-local_pt[1], T(0)) + std::max(local_pt[1] - length[1], T(0));
  T dz = local_pt[2];
  return dx * dx + dy * dy + dz * dz;
}

/// @brief Distance from point in local coordinates to rectangle
template <typename T>
auto local_point_rectangle_distance(const std::array<T, 3> &local_pt,
                                    const std::array<T, 2> &length) -> T {
  return tf::sqrt(local_point_rectangle_distance2(local_pt, length));
}

/// @brief Squared distance from point in local coordinates to segment
/// Segment [0, length] along x-axis at y=0 in 2D
template <typename T>
auto local_point_segment_distance2(const std::array<T, 2> &local_pt,
                                   T length) -> T {
  T dx = std::max(-local_pt[0], T(0)) + std::max(local_pt[0] - length, T(0));
  T dy = local_pt[1];
  return dx * dx + dy * dy;
}

/// @brief Distance from point in local coordinates to segment
template <typename T>
auto local_point_segment_distance(const std::array<T, 2> &local_pt,
                                  T length) -> T {
  return tf::sqrt(local_point_segment_distance2(local_pt, length));
}

/// @brief Squared distance between two segments in local coordinates (2D)
/// Segment A: [0, a] along local x-axis
/// Segment B: at translation t_ab, rotated by r_ab (2x2 rotation matrix)
/// r_ab[i][j] = dot(axis_a[i], axis_b[j])
template <typename T>
auto local_segment_distance2(const std::array<std::array<T, 2>, 2> &r_ab,
                             const std::array<T, 2> &t_ab,
                             T a, T b) -> T {
  // Segment A: from (0,0) to (a,0) in frame A
  // Segment B: from t_ab to t_ab + b * r_ab[*][0] in frame A

  // Direction of B in frame A: (r_ab[0][0], r_ab[1][0])
  T a_dot_b = r_ab[0][0];  // dot(axis_a0, axis_b0)
  T a_dot_t = t_ab[0];     // projection of translation onto axis_a0
  T b_dot_t = -t_ab[0] * r_ab[0][0] - t_ab[1] * r_ab[1][0];  // t_ba[0] in B's frame

  T t, u;
  impl::segment_closest_params(t, u, a, b, a_dot_b, a_dot_t, b_dot_t);

  // Closest points:
  // On A: (t, 0)
  // On B in frame A: (t_ab[0] + u * r_ab[0][0], t_ab[1] + u * r_ab[1][0])
  T dx = t_ab[0] + u * r_ab[0][0] - t;
  T dy = t_ab[1] + u * r_ab[1][0];
  return dx * dx + dy * dy;
}

/// @brief Distance between two segments in local coordinates (2D)
template <typename T>
auto local_segment_distance(const std::array<std::array<T, 2>, 2> &r_ab,
                            const std::array<T, 2> &t_ab,
                            T a, T b) -> T {
  return tf::sqrt(local_segment_distance2(r_ab, t_ab, a, b));
}

} // namespace tf::core
