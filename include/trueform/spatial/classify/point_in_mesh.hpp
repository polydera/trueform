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
#include "../../core/containment.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/distance.hpp"
#include "../../core/epsilon_inverse.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/intersects.hpp"
#include "../../core/ray_bv_check.hpp"
#include "../../core/polygons.hpp"
#include "../../core/ray.hpp"
#include "../../core/ray_hit.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/transformed.hpp"
#include "../../random/random_vector.hpp"
#include "../../topology/set_component_labels.hpp"
#include "../tree_search/intersects_bv.hpp"
#include "../tree_search/search.hpp"

namespace tf::spatial {
template <typename Policy0, typename Policy1, typename LabelType, typename F>
auto classify_point(const tf::point_like<3, Policy0> &_point,
                    const tf::polygons<Policy1> &polygons,
                    const tf::set_component_labels<LabelType> &scl,
                    const F &ignore_component) {
  auto frame = tf::frame_of(polygons);
  auto point = tf::transformed(_point, frame.inverse_transformation());
  const auto &tree = polygons.tree();
  if (!tf::spatial::intersects_bv(tree, point))
    return tf::containment::outside;
  using index_t = typename Policy1::index_type;
  using real_type = tf::coordinate_type<Policy0, Policy1>;
  struct node_t {
    index_t element;
    real_type t;
    tf::unit_vector<real_type, 3> normal;
  };

  tf::small_vector<node_t, 32> buffer;
  tf::ray<real_type, 3> ray;
  ray.origin = point;
  tf::vector<real_type, 3> ray_inv_dir;
  //
  auto is_on_edge = [](const auto &polygon, const auto &pt) {
    auto size = polygon.size();
    auto curr = size - 1;
    for (decltype(curr) next = 0; next < size; curr = next++) {
      if (tf::distance2(
              tf::make_segment_between_points(polygon[curr], polygon[next]),
              pt) < tf::epsilon2<real_type>)
        return true;
    }
    return false;
  };
  //
  bool failed = true;
  while (failed) {
    failed = false;
    buffer.clear();
    ray.direction = tf::random_vector<real_type, 3>();
    for (int i = 0; i < 3; ++i)
      ray_inv_dir[i] = tf::epsilon_inverse(ray.direction[i]);
    tf::spatial::search(
        tree,
        [&](const auto &bv) {
          real_type t0, t1;
          return tf::core::ray_bv_check(
                     ray, ray_inv_dir, bv, t0, t1, real_type(0),
                     std::numeric_limits<real_type>::max()) ==
                 tf::intersect_status::intersection;
        },
        [&](index_t poly_id) {
          if (ignore_component(scl.component_labels.labels[poly_id]))
            return false;
          auto polygon = polygons[poly_id] | tf::tag_plane();
          auto res = tf::ray_hit(ray, polygon);
          if (!res)
            return false;
          if (res.t < tf::epsilon<real_type>)
            return false;  // skip self-hit at origin
          if (is_on_edge(polygon, res.point)) {
            failed = true;
            return true;
          } else if (scl.set_types[scl.component_labels.labels[poly_id]] ==
                     tf::set_type::closed) {
            buffer.push_back({poly_id, res.t, polygon.normal()});
          }
          return false;
        });
  }
  if (!buffer.size())
    return tf::containment::outside;
  std::sort(buffer.begin(), buffer.end(),
            [](const auto &x, const auto &y) { return x.t < y.t; });
  tf::small_vector<int, 10> counts;
  counts.resize(scl.component_labels.n_components, 0);
  for (const auto &e : buffer) {
    if (tf::dot(e.normal, ray.direction) < 0)
      counts[scl.component_labels.labels[e.element]]++;
    else
      counts[scl.component_labels.labels[e.element]]--;
  }
  for (const auto &e : buffer) {
    if (counts[scl.component_labels.labels[e.element]] < 0)
      return tf::containment::inside;
    else if (counts[scl.component_labels.labels[e.element]] > 0)
      return tf::containment::outside;
  }
  return tf::containment::outside;
}

template <typename Policy0, typename Policy1, typename LabelType>
auto classify_point(const tf::point_like<3, Policy0> &_point,
                    const tf::polygons<Policy1> &polygons,
                    const tf::set_component_labels<LabelType> &scl) {
  return classify_point(_point, polygons, scl,
                        [](const auto &) { return false; });
}

template <typename Policy0, typename Policy1>
auto classify_point(const tf::point_like<3, Policy0> &_point,
                    const tf::polygons<Policy1> &polygons) {
  auto frame = tf::frame_of(polygons);
  auto point = tf::transformed(_point, frame.inverse_transformation());
  const auto &tree = polygons.tree();
  if (!tf::spatial::intersects_bv(tree, point))
    return tf::containment::outside;
  using index_t = typename Policy1::index_type;
  using real_type = tf::coordinate_type<Policy0, Policy1>;

  tf::ray<real_type, 3> ray;
  ray.origin = point;
  tf::vector<real_type, 3> ray_inv_dir;
  //
  auto is_on_edge = [](const auto &polygon, const auto &pt) {
    auto size = polygon.size();
    auto curr = size - 1;
    for (decltype(curr) next = 0; next < size; curr = next++) {
      if (tf::distance2(
              tf::make_segment_between_points(polygon[curr], polygon[next]),
              pt) < tf::epsilon2<real_type>)
        return true;
    }
    return false;
  };
  //
  bool failed = true;
  int count = 0;
  while (failed) {
    failed = false;
    count = 0;
    ray.direction = tf::random_vector<real_type, 3>();
    for (int i = 0; i < 3; ++i)
      ray_inv_dir[i] = tf::epsilon_inverse(ray.direction[i]);
    tf::spatial::search(
        tree,
        [&](const auto &bv) {
          real_type t0, t1;
          return tf::core::ray_bv_check(
                     ray, ray_inv_dir, bv, t0, t1, real_type(0),
                     std::numeric_limits<real_type>::max()) ==
                 tf::intersect_status::intersection;
        },
        [&](index_t poly_id) {
          auto polygon = polygons[poly_id] | tf::tag_plane();
          auto res = tf::ray_hit(ray, polygon);
          if (!res)
            return false;
          if (res.t < tf::epsilon<real_type>)
            return false;  // skip self-hit at origin
          if (is_on_edge(polygon, res.point)) {
            failed = true;
            return true;
          } else {
            if (tf::dot(polygon.normal, ray.direction) < 0)
              ++count;
            else
              --count;
          }
          return false;
        });
  }
  if (count < 0)
    return tf::containment::inside;
  else
    return tf::containment::outside;
}

} // namespace tf::spatial
