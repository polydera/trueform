/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./containment.hpp"
#include "./contains_point.hpp"
#include "./dot.hpp"
#include "./line_like.hpp"
#include "./plane_like.hpp"
#include "./point_like.hpp"
#include "./ray_like.hpp"
#include "./segment.hpp"
#include "./sidedness.hpp"
#include "./wedge.hpp"
namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto classify(const point_like<Dims, Policy0> &pt,
              const plane_like<Dims, Policy1> &pl) -> sidedness {
  auto d = tf::dot(pt, pl.normal) + pl.d;
  if (std::abs(d) < std::numeric_limits<decltype(d)>::epsilon())
    return sidedness::on_boundary;
  // on_negative_side == 1
  return static_cast<tf::sidedness>(d < 0);
}

template <typename Policy0, typename Policy1>
auto classify(const point_like<2, Policy0> &point,
              const segment<2, Policy1> &seg) -> sidedness {
  auto ac = seg[0] - point;
  auto bc = seg[1] - point;
  auto test = ac[0] * bc[1] - ac[1] * bc[0];
  if (std::abs(test) < std::numeric_limits<decltype(test)>::epsilon())
    return sidedness::on_boundary;
  // on_negative_side == 1
  return static_cast<tf::sidedness>(test < 0);
}

template <typename Policy0, typename Policy1>
auto classify(const point_like<2, Policy0> &point,
              const line_like<2, Policy1> &line) -> sidedness {
  return classify(point, tf::make_segment_between_points(line.origin, line(1)));
}

template <typename Policy0, typename Policy1>
auto classify(const point_like<2, Policy0> &point,
              const ray_like<2, Policy1> &ray) -> sidedness {
  return classify(point, tf::make_segment_between_points(ray.origin, ray(1)));
}

template <typename Policy0, typename Policy1>
auto classify(const point_like<2, Policy0> &pt, const wedge<Policy1> &w)
    -> strict_containment {
  using tf::sidedness;

  const auto o0 = classify(w[0], make_segment_between_points(w[1], w[2]));
  const auto o1 = classify(w[0], make_segment_between_points(w[1], pt));
  const auto o2 = classify(w[0], make_segment_between_points(w[2], pt));

  const bool convex = o0 != sidedness::on_negative_side;

  const bool inside_convex =
      o1 == sidedness::on_positive_side && o2 == sidedness::on_positive_side;

  const bool inside_reflex =
      o1 != sidedness::on_negative_side || o2 != sidedness::on_negative_side;

  const bool inside = (convex && inside_convex) || (!convex && inside_reflex);

  return static_cast<strict_containment>(!inside);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto classify(const point_like<Dims, Policy0> &pt,
              const polygon<Dims, Policy1> &poly) -> containment {
  return core::contains_point(poly, pt);
}
} // namespace tf
