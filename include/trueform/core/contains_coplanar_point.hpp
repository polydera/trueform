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
#include "./coordinate_type.hpp"
#include "./epsilon.hpp"
#include "./point_like.hpp"
#include "./polygon.hpp"
#include "./projector.hpp"
#include "./static_size.hpp"
#include <cstddef>

namespace tf {
namespace core {
/**
 * @ingroup core_queries
 * @brief Checks whether a point lies inside a polygon or on its edge (projected
 * to 2D).
 *
 * Uses the winding number algorithm, optionally accounting for numerical
 * precision via an epsilon tolerance when determining if the point lies exactly
 * on an edge.
 *
 * @tparam V Number of vertices in the polygon.
 * @tparam Policy Polygon storage policy.
 * @tparam Dims Dimensionality of the input point.
 * @tparam T Scalar type of the input point.
 * @tparam F Projector function type.
 * @param poly Polygon to test against.
 * @param input_pt The input point to check.
 * @param projector Function to project input and polygon to 2D.
 * @return true if the point is inside or on the polygon boundary.
 */
template <std::size_t Dims, typename Policy, typename T, typename F>
auto contains_coplanar_point(const tf::polygon<Dims, Policy> &poly,
                             const point_like<Dims, T> &input_pt,
                             const tf::projector<F> &projector)
    -> tf::containment {
  static_assert(tf::static_size_v<decltype(projector(poly[0]))> == 2,
                "We must project into 2D");
  using real_t = tf::coordinate_type<T, decltype(poly[0][0])>;
  int winding_number = 0;
  auto n = poly.size();

  tf::point<real_t, 2> pt = projector(input_pt);

  auto check_edge = [&](const auto &a, const auto &b) {
    auto ab = b - a;
    auto ap = pt - a;
    auto dy = b[1] - a[1];

    if (a[1] <= pt[1]) {
      if (b[1] > pt[1] && (ab[0] * ap[1] - ap[0] * dy) > 0)
        ++winding_number;
    } else {
      if (b[1] <= pt[1] && (ab[0] * ap[1] - ap[0] * dy) < 0)
        --winding_number;
    }
  };

  tf::point<real_t, 2> first_pt = projector(poly[0]);
  tf::point<real_t, 2> prev_pt = first_pt;
  for (std::size_t i = 1; i < n; i++) {
    tf::point<real_t, 2> current_pt = projector(poly[i]);
    check_edge(prev_pt, current_pt);
    prev_pt = current_pt;
  }
  check_edge(prev_pt, first_pt);

  return winding_number != 0 ? containment::inside : containment::outside;
}
} // namespace core

/*template <std::size_t Dims, typename Policy, typename T, typename F>*/
/*auto contains_coplanar_point(const tf::polygon<Dims, Policy> &poly,*/
/*                             const point_like<Dims, T> &input_pt,*/
/*                             const tf::projector<F> &projector) -> bool {*/
/*  return core::contains_coplanar_point(poly, input_pt, projector) !=*/
/*         containment::outside;*/
/*}*/
/// @ingroup core_queries
/// @brief Overload for 2D input points without projection
/// @copydoc contains_coplanar_point

/*template <typename Policy, typename T>*/
/*auto contains_coplanar_point(const tf::polygon<2, Policy> &poly,*/
/*                             const point_like<2, T> &input_pt) -> bool {*/
/*  return tf::contains_coplanar_point(poly, input_pt,*/
/*                                     tf::make_identity_projector());*/
/*}*/

/// @ingroup core_queries
/// @brief Overload for N-dimensional input points using automatic projection
/// @copydoc contains_coplanar_point

/*template <std::size_t Dims, typename Policy, typename T>*/
/*auto contains_coplanar_point(const tf::polygon<Dims, Policy> &poly,*/
/*                             const point_like<Dims, T> &input_pt) -> bool {*/
/*  return tf::contains_coplanar_point(poly, input_pt,*/
/*                                     tf::make_simple_projector(poly));*/
/*}*/
namespace core {
/**
 * @ingroup core_queries
 * @brief Checks whether a point lies inside or on a polygon with tolerance.
 *
 * Uses an epsilon to account for numerical precision when determining
 * if a point lies exactly on a polygon edge (in 2D after projection).
 *
 * @tparam V Number of vertices in the polygon.
 * @tparam Policy Polygon policy.
 * @tparam Dims Point dimensionality.
 * @tparam T Point scalar type.
 * @tparam F Projector type.
 * @param poly Polygon.
 * @param input_pt Input point.
 * @param epsilon Tolerance threshold.
 * @param projector Function that maps the input into 2D.
 * @return true if the point is inside or on the polygon.
 */
template <std::size_t Dims, typename Policy, typename T, typename F>
auto contains_coplanar_point(const tf::polygon<Dims, Policy> &poly,
                             const point_like<Dims, T> &input_pt,
                             const tf::projector<F> &projector,
                             tf::coordinate_type<T, Policy> epsilon)
    -> containment {
  auto epsilon2 = epsilon * epsilon;
  static_assert(tf::static_size_v<decltype(projector(poly[0]))> == 2,
                "We must project into 2D");
  using real_t = tf::coordinate_type<T, decltype(poly[0][0])>;
  int winding_number = 0;
  auto n = poly.size();

  tf::point<real_t, 2> pt = projector(input_pt);

  auto check_edge = [&](const auto &a, const auto &b) {
    auto ab = b - a;
    auto ap = pt - a;

    auto area = ab[0] * ap[1] - ab[1] * ap[0];
    auto ab_len2 = ab[0] * ab[0] + ab[1] * ab[1];

    if (ab_len2 > real_t(0) && area * area <= ab_len2 * epsilon2) {
      auto dot_ap_ab = ap[0] * ab[0] + ap[1] * ab[1];
      auto t = dot_ap_ab / ab_len2;
      if (t >= -epsilon && t <= real_t(1) + epsilon)
        return containment::on_boundary;
    }

    if (a[1] <= pt[1]) {
      if (b[1] > pt[1] && area > 0)
        ++winding_number;
    } else {
      if (b[1] <= pt[1] && area < 0)
        --winding_number;
    }
    return containment::outside;
  };

  tf::point<real_t, 2> first_pt = projector(poly[0]);
  tf::point<real_t, 2> prev_pt = first_pt;
  for (std::size_t i = 1; i < n; i++) {
    tf::point<real_t, 2> current_pt = projector(poly[i]);
    if (check_edge(prev_pt, current_pt) == containment::on_boundary)
      return containment::on_boundary;
    prev_pt = current_pt;
  }
  if (check_edge(prev_pt, first_pt) == containment::on_boundary)
    return containment::on_boundary;
  return winding_number != 0 ? containment::inside : containment::outside;
}
} // namespace core
template <std::size_t Dims, typename Policy, typename T, typename F>
auto contains_coplanar_point(const tf::polygon<Dims, Policy> &poly,
                             const point_like<Dims, T> &input_pt,
                             const tf::projector<F> &projector,
                             tf::coordinate_type<T, Policy> epsilon) -> bool {
  return core::contains_coplanar_point(poly, input_pt, projector, epsilon) !=
         containment::outside;
}
/// @ingroup core_queries
/// @brief Overload with epsilon and no projector (2D input)
/// @copydoc contains_coplanar_point

template <typename Policy, typename T>
auto contains_coplanar_point(const tf::polygon<2, Policy> &poly,
                             const point_like<2, T> &input_pt) -> bool {
  return tf::contains_coplanar_point(
      poly, input_pt, tf::make_identity_projector(),
      tf::epsilon<tf::coordinate_type<T, Policy>>);
}

/// @ingroup core_queries
/// @brief Overload with epsilon and auto projection
/// @copydoc contains_coplanar_point

template <std::size_t Dims, typename Policy, typename T>
auto contains_coplanar_point(const tf::polygon<Dims, Policy> &poly,
                             const point_like<Dims, T> &input_pt) -> bool {
  return tf::contains_coplanar_point(
      poly, input_pt, tf::make_simple_projector(poly),
      tf::epsilon<tf::coordinate_type<T, Policy>>);
}
} // namespace tf
