/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./aabb_metrics.hpp"
#include "./form.hpp"
#include "./implementation/tree_closest_point_using_sort_by_level.hpp"
#include "./implementation/tree_metric_result.hpp"
#include "./implementation/tree_tree_proximity_parallel.hpp"
#include "./nearest_neighbors.hpp"
#include "./transformed.hpp"
#include "./tree.hpp"
#include "trueform/tree_metric_info.hpp"
#include "trueform/tree_metric_info_pair.hpp"

namespace tf {

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto nearness_search(const tf::tree<Index, RealT, N> &tree,
                     const F0 &aabb_metric, const F1 &closest_point_f) {
  using tree_metric_t =
      tf::tree_metric_info<Index, decltype(closest_point_f(Index(0)))>;
  tf::implementation::tree_metric_result<tree_metric_t> result{
      std::numeric_limits<RealT>::max()};
  tf::implementation::tree_closest_point_using_sort_by_level(
      tree.nodes(), tree.ids(), aabb_metric, closest_point_f, result);
  return result.info;
}

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto nearness_search(const tf::tree<Index, RealT, N> &tree0,
                     const tf::tree<Index, RealT, N> &tree1,
                     const F0 &aabb_metrics_f, const F1 &closest_points_f) {

  using tree_metric_t =
      tf::tree_metric_info_pair<Index,
                                decltype(closest_points_f(Index(0), Index(0)))>;
  tf::implementation::local_tree_metric_result<tree_metric_t> result{
      std::numeric_limits<RealT>::max()};
  tf::implementation::tree_tree_proximity_parallel(tree0, tree1, aabb_metrics_f,
                                                   closest_points_f, result);
  return result.info();
}

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto nearness_search(const tf::tree<Index, RealT, N> &tree,
                     const F0 &aabb_metric, const F1 &closest_point_f,
                     RealT radius) {
  using tree_metric_t =
      tf::tree_metric_info<Index, decltype(closest_point_f(Index(0)))>;
  tf::implementation::tree_metric_result<tree_metric_t> result{radius * radius};
  tf::implementation::tree_closest_point_using_sort_by_level(
      tree.nodes(), tree.ids(), aabb_metric, closest_point_f, result);
  return result.info;
}
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto nearness_search(const tf::tree<Index, RealT, N> &tree0,
                     const tf::tree<Index, RealT, N> &tree1,
                     const F0 &aabb_metrics_f, const F1 &closest_points_f,
                     RealT radius) {
  using tree_metric_t =
      tf::tree_metric_info_pair<Index,
                                decltype(closest_points_f(Index(0), Index(0)))>;
  tf::implementation::local_tree_metric_result<tree_metric_t> result{radius *
                                                                     radius};
  tf::implementation::tree_tree_proximity_parallel(tree0, tree1, aabb_metrics_f,
                                                   closest_points_f, result);
  return result.info();
}

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1, typename RandomIt>
auto nearness_search(const tf::tree<Index, RealT, N> &tree,
                     const F0 &aabb_metric, const F1 &closest_point_f,
                     tf::nearest_neighbors<RandomIt> &knn) {
  tf::implementation::tree_closest_point_using_sort_by_level(
      tree.nodes(), tree.ids(), aabb_metric, closest_point_f, knn);
}

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1, typename RandomIt>
auto nearness_search(const tf::tree<Index, RealT, N> &tree,
                     const F0 &aabb_metric, const F1 &closest_point_f,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  tf::implementation::tree_closest_point_using_sort_by_level(
      tree.nodes(), tree.ids(), aabb_metric, closest_point_f, knn);
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &aabb_metric,
                     const F1 &closest_point_f) {
  return nearness_search(
      form.tree(),
      [&](const auto &aabb) {
        return aabb_metric(tf::transformed(aabb, form.transformation()));
      },
      [&](auto id) {
        return closest_point_f(
            tf::transformed(form[id], form.transformation()));
      });
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &aabb_metric,
                     const F1 &closest_point_f,
                     typename Policy0::real_t radius) {
  return nearness_search(
      form.tree(),
      [&](const auto &aabb) {
        return aabb_metric(tf::transformed(aabb, form.transformation()));
      },
      [&](auto id) {
        return closest_point_f(
            tf::transformed(form[id], form.transformation()));
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1,
          typename RandomIt>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &aabb_metric,
                     const F1 &closest_point_f,
                     nearest_neighbors<RandomIt> &knn) {
  return nearness_search(
      form.tree(),
      [&](const auto &aabb) {
        return aabb_metric(tf::transformed(aabb, form.transformation()));
      },
      [&](auto id) {
        return closest_point_f(
            tf::transformed(form[id], form.transformation()));
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1,
          typename RandomIt>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &aabb_metric,
                     const F1 &closest_point_f,
                     nearest_neighbors<RandomIt> &&knn) {
  return nearness_search(form, aabb_metric, closest_point_f, knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename F>
auto nearness_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1,
                     const F &closest_point_f) {
  return nearness_search(
      form0.tree(), form1.tree(),
      [&](const auto &aabb0, const auto &aabb1) {
        return tf::make_aabb_metrics(
            tf::transformed(aabb0, form0.transformation()),
            tf::transformed(aabb1, form1.transformation()));
      },
      [&](auto id0, auto id1) {
        return closest_point_f(
            tf::transformed(form0[id0], form0.transformation()),
            tf::transformed(form1[id1], form1.transformation()));
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename F>
auto nearness_search(
    const tf::form<Dims, Policy0> &form0, const tf::form<Dims, Policy1> &form1,
    const F &closest_point_f,
    tf::common_value<typename Policy0::real_t, typename Policy1::real_t>
        radius) {
  return nearness_search(
      form0.tree(), form1.tree(),
      [&](const auto &aabb0, const auto &aabb1) {
        return tf::make_aabb_metrics(
            tf::transformed(aabb0, form0.transformation()),
            tf::transformed(aabb1, form1.transformation()));
      },
      [&](auto id0, auto id1) {
        return closest_point_f(
            tf::transformed(form0[id0], form0.transformation()),
            tf::transformed(form1[id1], form1.transformation()));
      },
      radius);
}
} // namespace tf
