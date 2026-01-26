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

/// @ingroup core_queries
/// @brief Classify a point's position relative to a plane.
///
/// Determines whether the point lies on the positive side, negative side,
/// or on the boundary of the plane.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy0 The point policy.
/// @tparam Policy1 The plane policy.
/// @param pt The point to classify.
/// @param pl The plane to test against.
/// @return A @ref tf::sidedness value indicating the point's position.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto classify(const point_like<Dims, Policy0> &pt,
              const plane_like<Dims, Policy1> &pl) -> sidedness {
  auto d = tf::dot(pt, pl.normal) + pl.d;
  if (std::abs(d) < tf::epsilon<decltype(d)>)
    return sidedness::on_boundary;
  // on_negative_side == 1
  return static_cast<tf::sidedness>(d < 0);
}

/// @ingroup core_queries
/// @brief Classify a 2D point's position relative to a segment.
///
/// Determines whether the point lies on the left (positive), right (negative),
/// or on the segment itself (boundary).
///
/// @tparam Policy0 The point policy.
/// @tparam Policy1 The segment policy.
/// @param point The point to classify.
/// @param seg The segment to test against.
/// @return A @ref tf::sidedness value indicating the point's position.
template <typename Policy0, typename Policy1>
auto classify(const tf::point_like<2, Policy0> &point,
              const tf::segment<2, Policy1> &seg) -> tf::sidedness {
  using real = decltype(seg[0][0] * seg[1][1]);

  const auto ab = seg[1] - seg[0];
  const auto ap = point - seg[0];

  const real test = ab[0] * ap[1] - ab[1] * ap[0];
  const real eps = tf::epsilon2<real>;

  if (test > eps)
    return tf::sidedness::on_positive_side;
  if (test < -eps)
    return tf::sidedness::on_negative_side;

  const real t = ap[0] * ab[0] + ap[1] * ab[1];
  const real ab2 = ab[0] * ab[0] + ab[1] * ab[1];
  if (t >= -eps && t <= ab2 + eps)
    return tf::sidedness::on_boundary;

  return tf::sidedness::on_negative_side;
}

/// @ingroup core_queries
/// @brief Classify a 2D point's position relative to a line.
///
/// Determines whether the point lies on the left (positive), right (negative),
/// or on the line itself (boundary).
///
/// @tparam Policy0 The point policy.
/// @tparam Policy1 The line policy.
/// @param point The point to classify.
/// @param line The line to test against.
/// @return A @ref tf::sidedness value indicating the point's position.
template <typename Policy0, typename Policy1>
auto classify(const tf::point_like<2, Policy0> &point,
              const tf::line_like<2, Policy1> &line) -> tf::sidedness {
  using real = decltype(point[0] * line.origin[0]);
  const auto dir = line.direction;
  const auto ap = point - line.origin;

  const real test = dir[0] * ap[1] - dir[1] * ap[0];
  const real eps = tf::epsilon2<real>;

  if (test > eps)
    return tf::sidedness::on_positive_side;
  if (test < -eps)
    return tf::sidedness::on_negative_side;

  return tf::sidedness::on_boundary;
}

/// @ingroup core_queries
/// @brief Classify a 2D point's position relative to a ray.
///
/// Determines whether the point lies on the left (positive), right (negative),
/// or on the ray itself (boundary).
///
/// @tparam Policy0 The point policy.
/// @tparam Policy1 The ray policy.
/// @param point The point to classify.
/// @param ray The ray to test against.
/// @return A @ref tf::sidedness value indicating the point's position.
template <typename Policy0, typename Policy1>
auto classify(const tf::point_like<2, Policy0> &point,
              const tf::ray_like<2, Policy1> &ray) -> tf::sidedness {
  using real = decltype(point[0] * ray.origin[0]);
  const auto dir = ray.direction;
  const auto ap = point - ray.origin;

  const real test = dir[0] * ap[1] - dir[1] * ap[0];
  const real eps = tf::epsilon2<real>;

  if (test > eps)
    return tf::sidedness::on_positive_side;
  if (test < -eps)
    return tf::sidedness::on_negative_side;

  const real t = ap[0] * dir[0] + ap[1] * dir[1];
  if (t >= -eps)
    return tf::sidedness::on_boundary;

  return tf::sidedness::on_negative_side;
}

/// @ingroup core_queries
/// @brief Classify a 2D point's position relative to a wedge.
///
/// Determines whether the point lies inside or outside the wedge.
///
/// @tparam Policy0 The point policy.
/// @tparam Policy1 The wedge policy.
/// @param pt The point to classify.
/// @param w The wedge to test against.
/// @return A @ref tf::strict_containment value.
template <typename Policy0, typename Policy1>
auto classify(const tf::point_like<2, Policy0> &pt, const tf::wedge<Policy1> &w)
    -> strict_containment {
  using tf::sidedness;

  // o0 = orient(O,A,B)
  const auto o0 = classify(w[2], tf::make_segment_between_points(w[0], w[1]));
  // o1 = orient(O,A,pt)
  const auto o1 = classify(pt, tf::make_segment_between_points(w[0], w[1]));
  // o2 = orient(A,B,pt)
  const auto o2 = classify(pt, tf::make_segment_between_points(w[1], w[2]));

  const bool convex = (o0 != sidedness::on_negative_side);

  // Convex: include boundary (>=0)
  const bool inside_convex = (o1 != sidedness::on_negative_side) &&
                             (o2 != sidedness::on_negative_side);

  // Reflex: exclude boundary (strict >0)
  const bool inside_reflex = (o1 == sidedness::on_positive_side) ||
                             (o2 == sidedness::on_positive_side);

  const bool inside = (convex && inside_convex) || (!convex && inside_reflex);
  return static_cast<strict_containment>(!inside);
}

/// @ingroup core_queries
/// @brief Classify a point's position relative to a polygon.
///
/// Determines whether the point lies inside, outside, or on the boundary
/// of the polygon.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy0 The point policy.
/// @tparam Policy1 The polygon policy.
/// @param pt The point to classify.
/// @param poly The polygon to test against.
/// @return A @ref tf::containment value.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto classify(const point_like<Dims, Policy0> &pt,
              const polygon<Dims, Policy1> &poly) -> containment {
  return core::contains_point(poly, pt);
}
} // namespace tf
