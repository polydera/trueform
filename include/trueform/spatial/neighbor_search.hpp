/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/aabb_from.hpp"
#include "../core/closest_metric_point_pair.hpp"
#include "../core/line_like.hpp"
#include "../core/polygon.hpp"
#include "../core/ray_like.hpp"
#include "../core/segment.hpp"
#include "./search/neighbor_search.hpp"

namespace tf {

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj) {
  return spatial::neighbor_search(form, obj, obj);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     typename Policy0::real_t radius) {
  return spatial::neighbor_search(form, obj, obj, radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return spatial::neighbor_search(form, obj, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return spatial::neighbor_search(form, obj, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj) {
  return spatial::admissible_neighbor_search(
      form, obj, std::numeric_limits<typename Policy0::real_t>::max());
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     typename Policy0::real_t radius) {
  return spatial::admissible_neighbor_search(form, obj, radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return spatial::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return spatial::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj) {
  return spatial::admissible_neighbor_search(
      form, obj, std::numeric_limits<typename Policy0::real_t>::max());
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     typename Policy0::real_t radius) {
  return spatial::admissible_neighbor_search(form, obj, radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return spatial::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return spatial::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj) {
  return spatial::admissible_neighbor_search(
      form, obj, std::numeric_limits<typename Policy0::real_t>::max());
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     typename Policy0::real_t radius) {
  return spatial::admissible_neighbor_search(form, obj, radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return spatial::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return spatial::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj) {
  return spatial::neighbor_search(form, tf::aabb_from(obj), tf::tag_plane(obj));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     typename Policy0::real_t radius) {
  return spatial::neighbor_search(form, tf::aabb_from(obj), tf::tag_plane(obj),
                                  radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return spatial::neighbor_search(form, tf::aabb_from(obj), tf::tag_plane(obj),
                                  knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return spatial::neighbor_search(form, tf::aabb_from(obj), tf::tag_plane(obj),
                                  knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1) {
  return tf::nearness_search(form0, form1,
                             [](const auto &obj0, const auto &obj1) {
                               return tf::closest_metric_point_pair(obj0, obj1);
                             });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1,
                     typename Policy0::real_t radius) {
  return tf::nearness_search(
      form0, form1,
      [](const auto &obj0, const auto &obj1) {
        return tf::closest_metric_point_pair(obj0, obj1);
      },
      radius);
}

} // namespace tf
