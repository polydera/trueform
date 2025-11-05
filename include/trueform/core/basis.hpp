/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./cross.hpp"
#include "./dot.hpp"
#include "./plane_like.hpp"
#include "./point_like.hpp"
#include "./unit_vector.hpp"
#include "./vector.hpp"

namespace tf {
template <typename Policy> auto make_basis(const plane_like<3, Policy> &plane) {
  const auto &normal = plane.normal;
  tf::vector<tf::coordinate_type<Policy>, 3> t0;
  if (std::abs(normal[0]) < std::abs(normal[1])) {
    t0[0] = 0;
    t0[1] = -normal[2];
    t0[2] = normal[1];
  } else {
    t0[0] = normal[2];
    t0[1] = 0;
    t0[2] = -normal[0];
  }
  auto t1 = tf::cross(normal, t0);
  return std::array<tf::unit_vector<tf::coordinate_type<Policy>, 3>, 2>{t0, t1};
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename Policy2>
auto make_basis(const tf::point_like<Dims, Policy0> &p0,
                const tf::point_like<Dims, Policy1> &p1,
                const tf::point_like<Dims, Policy2> &p2) {
  using real_t = tf::coordinate_type<Policy0, Policy1, Policy2>;
  tf::unit_vector<real_t, Dims> e0 = p1 - p0;
  auto e1 = p2 - p0;
  auto cd0 = tf::dot(e0, e1) * e0;
  e1 -= cd0;
  return std::array<tf::unit_vector<real_t, Dims>, 2>{e0, e1};
}
} // namespace tf
