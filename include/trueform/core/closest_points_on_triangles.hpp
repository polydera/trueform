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
#include "./cross.hpp"
#include "./dot.hpp"
#include "./epsilon.hpp"
#include "./metric_point_pair.hpp"
#include "./point.hpp"
#include "./polygon.hpp"
#include "./vector.hpp"

namespace tf::core {

template <typename T>
void triangle_segment_closest_points(const tf::point<T, 3> &p,
                                     const tf::vector<T, 3> &a,
                                     const tf::point<T, 3> &q,
                                     const tf::vector<T, 3> &b,
                                     tf::vector<T, 3> &vec, tf::point<T, 3> &x,
                                     tf::point<T, 3> &y) {
  tf::vector<T, 3> t_vec = q - p;
  T a_dot_a = tf::dot(a, a);
  T b_dot_b = tf::dot(b, b);
  T a_dot_b = tf::dot(a, b);
  T a_dot_t = tf::dot(a, t_vec);
  T b_dot_t = tf::dot(b, t_vec);

  T denom = a_dot_a * b_dot_b - a_dot_b * a_dot_b;
  T t =
      (denom != T(0)) ? (a_dot_t * b_dot_b - b_dot_t * a_dot_b) / denom : T(0);

  if (t < T(0) || std::isnan(t))
    t = T(0);
  else if (t > T(1))
    t = T(1);

  T u = (t * a_dot_b - b_dot_t) / b_dot_b;

  if (u <= T(0) || std::isnan(u)) {
    y = q;
    t = a_dot_t / a_dot_a;
    if (t <= T(0) || std::isnan(t)) {
      x = p;
      vec = q - p;
    } else if (t >= T(1)) {
      x = p + a;
      vec = q - x;
    } else {
      x = p + a * t;
      auto tmp = tf::cross(t_vec, a);
      vec = tf::cross(a, tmp);
    }
  } else if (u >= T(1)) {
    y = q + b;
    t = (a_dot_b + a_dot_t) / a_dot_a;
    if (t <= T(0) || std::isnan(t)) {
      x = p;
      vec = y - p;
    } else if (t >= T(1)) {
      x = p + a;
      vec = y - x;
    } else {
      x = p + a * t;
      auto t2 = y - p;
      auto tmp = tf::cross(t2, a);
      vec = tf::cross(a, tmp);
    }
  } else {
    y = q + b * u;
    if (t <= T(0) || std::isnan(t)) {
      x = p;
      auto tmp = tf::cross(t_vec, b);
      vec = tf::cross(b, tmp);
    } else if (t >= T(1)) {
      x = p + a;
      auto t2 = q - x;
      auto tmp = tf::cross(t2, b);
      vec = tf::cross(b, tmp);
    } else {
      x = p + a * t;
      vec = tf::cross(a, b);
      if (tf::dot(vec, t_vec) < T(0)) {
        vec = vec * T(-1);
      }
    }
  }
}

template <typename Policy0, typename Policy1>
auto closest_points_on_triangles(const tf::polygon<3, Policy0> &tri1,
                                 const tf::polygon<3, Policy1> &tri2) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  using point_t = tf::point<T, 3>;
  using vec_t = tf::vector<T, 3>;

  // Edge vectors
  vec_t s_edges[3] = {tri1[1] - tri1[0], tri1[2] - tri1[1], tri1[0] - tri1[2]};
  vec_t t_edges[3] = {tri2[1] - tri2[0], tri2[2] - tri2[1], tri2[0] - tri2[2]};

  vec_t vec;
  point_t p, q;
  point_t min_p = tri1[0];
  point_t min_q = tri2[0];
  T min_dist2 = (tri1[0] - tri2[0]).length2() + T(1);
  int shown_disjoint = 0;

  // Test all 9 edge pairs
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      triangle_segment_closest_points(point_t(tri1[i]), s_edges[i],
                                      point_t(tri2[j]), t_edges[j], vec, p, q);
      vec_t v = q - p;
      T dd = tf::dot(v, v);

      if (dd <= min_dist2) {
        min_p = p;
        min_q = q;
        min_dist2 = dd;

        vec_t z = tri1[(i + 2) % 3] - p;
        T a = tf::dot(z, vec);
        z = tri2[(j + 2) % 3] - q;
        T b = tf::dot(z, vec);

        if ((a <= T(0)) && (b >= T(0)))
          return tf::make_metric_point_pair(dd, min_p, min_q);

        T proj = tf::dot(v, vec);
        if (a < T(0))
          a = T(0);
        if (b > T(0))
          b = T(0);
        if ((proj - a + b) > T(0))
          shown_disjoint = 1;
      }
    }
  }

  // Check vertex-to-face case for tri1
  vec_t s_normal = tf::cross(s_edges[0], s_edges[1]);
  T s_normal_len2 = tf::dot(s_normal, s_normal);

  if (s_normal_len2 > tf::epsilon2<T>) {
    vec_t t_proj;
    t_proj[0] = tf::dot(vec_t(tri1[0] - tri2[0]), s_normal);
    t_proj[1] = tf::dot(vec_t(tri1[0] - tri2[1]), s_normal);
    t_proj[2] = tf::dot(vec_t(tri1[0] - tri2[2]), s_normal);

    int point = -1;
    if ((t_proj[0] > T(0)) && (t_proj[1] > T(0)) && (t_proj[2] > T(0))) {
      point = (t_proj[0] < t_proj[1]) ? 0 : 1;
      if (t_proj[2] < t_proj[point])
        point = 2;
    } else if ((t_proj[0] < T(0)) && (t_proj[1] < T(0)) && (t_proj[2] < T(0))) {
      point = (t_proj[0] > t_proj[1]) ? 0 : 1;
      if (t_proj[2] > t_proj[point])
        point = 2;
    }

    if (point >= 0) {
      shown_disjoint = 1;
      vec_t v = tri2[point] - tri1[0];
      if (tf::dot(v, tf::cross(s_normal, s_edges[0])) > T(0)) {
        v = tri2[point] - tri1[1];
        if (tf::dot(v, tf::cross(s_normal, s_edges[1])) > T(0)) {
          v = tri2[point] - tri1[2];
          if (tf::dot(v, tf::cross(s_normal, s_edges[2])) > T(0)) {
            point_t p_out = point_t(tri2[point]) +
                            s_normal * (t_proj[point] / s_normal_len2);
            point_t q_out = tri2[point];
            return tf::make_metric_point_pair((p_out - q_out).length2(), p_out,
                                              q_out);
          }
        }
      }
    }
  }

  // Check vertex-to-face case for tri2
  vec_t t_normal = tf::cross(t_edges[0], t_edges[1]);
  T t_normal_len2 = tf::dot(t_normal, t_normal);

  if (t_normal_len2 > tf::epsilon2<T>) {
    vec_t s_proj;
    s_proj[0] = tf::dot(vec_t(tri2[0] - tri1[0]), t_normal);
    s_proj[1] = tf::dot(vec_t(tri2[0] - tri1[1]), t_normal);
    s_proj[2] = tf::dot(vec_t(tri2[0] - tri1[2]), t_normal);

    int point = -1;
    if ((s_proj[0] > T(0)) && (s_proj[1] > T(0)) && (s_proj[2] > T(0))) {
      point = (s_proj[0] < s_proj[1]) ? 0 : 1;
      if (s_proj[2] < s_proj[point])
        point = 2;
    } else if ((s_proj[0] < T(0)) && (s_proj[1] < T(0)) && (s_proj[2] < T(0))) {
      point = (s_proj[0] > s_proj[1]) ? 0 : 1;
      if (s_proj[2] > s_proj[point])
        point = 2;
    }

    if (point >= 0) {
      shown_disjoint = 1;
      vec_t v = tri1[point] - tri2[0];
      if (tf::dot(v, tf::cross(t_normal, t_edges[0])) > T(0)) {
        v = tri1[point] - tri2[1];
        if (tf::dot(v, tf::cross(t_normal, t_edges[1])) > T(0)) {
          v = tri1[point] - tri2[2];
          if (tf::dot(v, tf::cross(t_normal, t_edges[2])) > T(0)) {
            point_t p_out = tri1[point];
            point_t q_out = point_t(tri1[point]) +
                            t_normal * (s_proj[point] / t_normal_len2);
            return tf::make_metric_point_pair((p_out - q_out).length2(), p_out,
                                              q_out);
          }
        }
      }
    }
  }

  if (shown_disjoint) {
    return tf::make_metric_point_pair(min_dist2, min_p, min_q);
  }
  return tf::make_metric_point_pair(T(0), min_p, min_q);
}

} // namespace tf::core
