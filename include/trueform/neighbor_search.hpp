/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./aabb_from.hpp"
#include "./closest_metric_point_pair.hpp"
#include "./implementation/neighbor_search.hpp"
#include "./line.hpp"
#include "./polygon.hpp"
#include "./ray.hpp"
#include "./segment.hpp"

namespace tf {

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj) {
  return implementation::neighbor_search(form, obj, obj);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     typename Policy0::real_t radius) {
  return implementation::neighbor_search(form, obj, obj, radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return implementation::neighbor_search(form, obj, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return implementation::neighbor_search(form, obj, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Policy1> &obj) {
  return implementation::admissible_neighbor_search(
      form, obj, std::numeric_limits<typename Policy0::real_t>::max());
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Policy1> &obj,
                     typename Policy0::real_t radius) {
  return implementation::admissible_neighbor_search(form, obj, radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return implementation::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return implementation::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename RealT>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray<RealT, Dims> &obj) {
  return implementation::admissible_neighbor_search(
      form, obj, std::numeric_limits<typename Policy0::real_t>::max());
}

template <std::size_t Dims, typename Policy0, typename RealT>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray<RealT, Dims> &obj,
                     typename Policy0::real_t radius) {
  return implementation::admissible_neighbor_search(form, obj, radius);
}

template <std::size_t Dims, typename Policy0, typename RealT,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray<RealT, Dims> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return implementation::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename RealT,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray<RealT, Dims> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return implementation::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename RealT>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line<RealT, Dims> &obj) {
  return implementation::admissible_neighbor_search(
      form, obj, std::numeric_limits<typename Policy0::real_t>::max());
}

template <std::size_t Dims, typename Policy0, typename RealT>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line<RealT, Dims> &obj,
                     typename Policy0::real_t radius) {
  return implementation::admissible_neighbor_search(form, obj, radius);
}

template <std::size_t Dims, typename Policy0, typename RealT,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line<RealT, Dims> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return implementation::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, typename RealT,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line<RealT, Dims> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return implementation::admissible_neighbor_search(form, obj, knn);
}

template <std::size_t Dims, typename Policy0, std::size_t V, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<V, Policy1> &obj) {
  return implementation::neighbor_search(form, tf::aabb_from(obj),
                                         tf::inject_plane(obj));
}

template <std::size_t Dims, typename Policy0, std::size_t V, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<V, Policy1> &obj,
                     typename Policy0::real_t radius) {
  return implementation::neighbor_search(form, tf::aabb_from(obj),
                                         tf::inject_plane(obj), radius);
}

template <std::size_t Dims, typename Policy0, std::size_t V, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<V, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &knn) {
  return implementation::neighbor_search(form, tf::aabb_from(obj),
                                         tf::inject_plane(obj), knn);
}

template <std::size_t Dims, typename Policy0, std::size_t V, typename Policy1,
          typename RandomIter>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<V, Policy1> &obj,
                     tf::nearest_neighbors<RandomIter> &&knn) {
  return implementation::neighbor_search(form, tf::aabb_from(obj),
                                         tf::inject_plane(obj), knn);
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
