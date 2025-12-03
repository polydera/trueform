/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./angle.hpp"
#include "./axis.hpp"
#include "./point_like.hpp"
#include "./transformation.hpp"
#include "./unit_vector_like.hpp"

namespace tf {

template <typename T, typename Policy>
auto make_rotation(rad<T> angle, const unit_vector_like<3, Policy>& axis)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T x = axis[0], y = axis[1], z = axis[2];
  return transformation<T, 3>{
    t*x*x + c,     t*x*y - s*z,   t*x*z + s*y,   0,
    t*x*y + s*z,   t*y*y + c,     t*y*z - s*x,   0,
    t*x*z - s*y,   t*y*z + s*x,   t*z*z + c,     0
  };
}

template <typename T, typename Policy>
auto make_rotation(deg<T> angle, const unit_vector_like<3, Policy>& axis)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis);
}

template <typename T>
auto make_rotation(rad<T> angle, axis_t<0>) -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  return transformation<T, 3>{
    1, 0,  0, 0,
    0, c, -s, 0,
    0, s,  c, 0
  };
}

template <typename T>
auto make_rotation(deg<T> angle, axis_t<0>) -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<0>{});
}

template <typename T>
auto make_rotation(rad<T> angle, axis_t<1>) -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  return transformation<T, 3>{
     c, 0, s, 0,
     0, 1, 0, 0,
    -s, 0, c, 0
  };
}

template <typename T>
auto make_rotation(deg<T> angle, axis_t<1>) -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<1>{});
}

template <typename T>
auto make_rotation(rad<T> angle, axis_t<2>) -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  return transformation<T, 3>{
    c, -s, 0, 0,
    s,  c, 0, 0,
    0,  0, 1, 0
  };
}

template <typename T>
auto make_rotation(deg<T> angle, axis_t<2>) -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<2>{});
}

template <typename T>
auto make_rotation(rad<T> angle) -> transformation<T, 2> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  return transformation<T, 2>{
    c, -s, 0,
    s,  c, 0
  };
}

template <typename T>
auto make_rotation(deg<T> angle) -> transformation<T, 2> {
  return make_rotation(rad<T>{angle});
}

template <typename T, typename AxisPolicy, typename PointPolicy>
auto make_rotation(rad<T> angle, const unit_vector_like<3, AxisPolicy>& axis,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T x = axis[0], y = axis[1], z = axis[2];
  T px = pivot[0], py = pivot[1], pz = pivot[2];
  T r00 = t*x*x + c,   r01 = t*x*y - s*z, r02 = t*x*z + s*y;
  T r10 = t*x*y + s*z, r11 = t*y*y + c,   r12 = t*y*z - s*x;
  T r20 = t*x*z - s*y, r21 = t*y*z + s*x, r22 = t*z*z + c;
  return transformation<T, 3>{
    r00, r01, r02, px - (r00*px + r01*py + r02*pz),
    r10, r11, r12, py - (r10*px + r11*py + r12*pz),
    r20, r21, r22, pz - (r20*px + r21*py + r22*pz)
  };
}

template <typename T, typename AxisPolicy, typename PointPolicy>
auto make_rotation(deg<T> angle, const unit_vector_like<3, AxisPolicy>& axis,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis, pivot);
}

template <typename T, typename PointPolicy>
auto make_rotation(rad<T> angle, axis_t<0>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T py = pivot[1], pz = pivot[2];
  return transformation<T, 3>{
    1, 0,  0, 0,
    0, c, -s, py*t + s*pz,
    0, s,  c, pz*t - s*py
  };
}

template <typename T, typename PointPolicy>
auto make_rotation(deg<T> angle, axis_t<0>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<0>{}, pivot);
}

template <typename T, typename PointPolicy>
auto make_rotation(rad<T> angle, axis_t<1>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T px = pivot[0], pz = pivot[2];
  return transformation<T, 3>{
     c, 0, s, px*t - s*pz,
     0, 1, 0, 0,
    -s, 0, c, pz*t + s*px
  };
}

template <typename T, typename PointPolicy>
auto make_rotation(deg<T> angle, axis_t<1>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<1>{}, pivot);
}

template <typename T, typename PointPolicy>
auto make_rotation(rad<T> angle, axis_t<2>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T px = pivot[0], py = pivot[1];
  return transformation<T, 3>{
    c, -s, 0, px*t + s*py,
    s,  c, 0, py*t - s*px,
    0,  0, 1, 0
  };
}

template <typename T, typename PointPolicy>
auto make_rotation(deg<T> angle, axis_t<2>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<2>{}, pivot);
}

template <typename T, typename PointPolicy>
auto make_rotation(rad<T> angle, const point_like<2, PointPolicy>& pivot)
    -> transformation<T, 2> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T px = pivot[0], py = pivot[1];
  return transformation<T, 2>{
    c, -s, px*t + s*py,
    s,  c, py*t - s*px
  };
}

template <typename T, typename PointPolicy>
auto make_rotation(deg<T> angle, const point_like<2, PointPolicy>& pivot)
    -> transformation<T, 2> {
  return make_rotation(rad<T>{angle}, pivot);
}

} // namespace tf
