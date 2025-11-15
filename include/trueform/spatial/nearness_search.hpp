/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/aabb_metrics.hpp"
#include "../core/transformed.hpp"
#include "./form.hpp"
#include "./nearest_neighbors.hpp"
#include "./search/tree_metric_result.hpp"
#include "./search/tree_proximity.hpp"
#include "./search/tree_tree_proximity.hpp"
#include "./tree.hpp"
#include "./tree_metric_info.hpp"
#include "./tree_metric_info_pair.hpp"

namespace tf {

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto nearness_search(const tf::tree<Index, RealT, N> &tree,
                     const F0 &aabb_metric, const F1 &closest_point_f) {
  using tree_metric_t =
      tf::tree_metric_info<Index, decltype(closest_point_f(Index(0)))>;
  tf::spatial::tree_metric_result<tree_metric_t> result{
      std::numeric_limits<RealT>::max()};
  tf::spatial::tree_proximity(tree.nodes(), tree.ids(), aabb_metric,
                              closest_point_f, result);
  return result.info;
}

template <typename Index0, typename Index1, typename RealT, std::size_t N,
          typename F0, typename F1>
auto nearness_search(const tf::tree<Index0, RealT, N> &tree0,
                     const tf::tree<Index1, RealT, N> &tree1,
                     const F0 &aabb_metrics_f, const F1 &closest_points_f) {

  using tree_metric_t = tf::tree_metric_info_pair<
      Index0, Index1, decltype(closest_points_f(Index0(0), Index1(0)))>;
  tf::spatial::local_tree_metric_result<tree_metric_t> result{
      std::numeric_limits<RealT>::max()};
  tf::spatial::tree_tree_proximity(tree0, tree1, aabb_metrics_f,
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
  tf::spatial::tree_metric_result<tree_metric_t> result{radius * radius};
  tf::spatial::tree_proximity(tree.nodes(), tree.ids(), aabb_metric,
                              closest_point_f, result);
  return result.info;
}
template <typename Index0, typename Index1, typename RealT, std::size_t N,
          typename F0, typename F1>
auto nearness_search(const tf::tree<Index0, RealT, N> &tree0,
                     const tf::tree<Index1, RealT, N> &tree1,
                     const F0 &aabb_metrics_f, const F1 &closest_points_f,
                     RealT radius) {
  using tree_metric_t = tf::tree_metric_info_pair<
      Index0, Index1, decltype(closest_points_f(Index0(0), Index1(0)))>;
  tf::spatial::local_tree_metric_result<tree_metric_t> result{radius * radius};
  tf::spatial::tree_tree_proximity(tree0, tree1, aabb_metrics_f,
                                   closest_points_f, result);
  return result.info();
}

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1, typename RandomIt>
auto nearness_search(const tf::tree<Index, RealT, N> &tree,
                     const F0 &aabb_metric, const F1 &closest_point_f,
                     tf::nearest_neighbors<RandomIt> &knn)
    -> tf::nearest_neighbors<RandomIt> & {
  tf::spatial::tree_proximity(tree.nodes(), tree.ids(), aabb_metric,
                              closest_point_f, knn);
  return knn;
}

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1, typename RandomIt>
auto nearness_search(const tf::tree<Index, RealT, N> &tree,
                     const F0 &aabb_metric, const F1 &closest_point_f,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  tf::spatial::tree_proximity(tree.nodes(), tree.ids(), aabb_metric,
                              closest_point_f, knn);
  return knn;
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &aabb_metric,
                     const F1 &closest_point_f) {
  return nearness_search(
      form.tree(),
      [&](const auto &aabb) {
        return aabb_metric(tf::transformed(aabb, form.frame()));
      },
      [&](auto id) {
        return closest_point_f(tf::transformed(form[id], form.frame()));
      });
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &aabb_metric,
                     const F1 &closest_point_f,
                     typename Policy0::real_t radius) {
  return nearness_search(
      form.tree(),
      [&](const auto &aabb) {
        return aabb_metric(tf::transformed(aabb, form.frame()));
      },
      [&](auto id) {
        return closest_point_f(tf::transformed(form[id], form.frame()));
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1,
          typename RandomIt>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &aabb_metric,
                     const F1 &closest_point_f,
                     nearest_neighbors<RandomIt> &knn)
    -> tf::nearest_neighbors<RandomIt> & {
  nearness_search(
      form.tree(),
      [&](const auto &aabb) {
        return aabb_metric(tf::transformed(aabb, form.frame()));
      },
      [&](auto id) {
        return closest_point_f(tf::transformed(form[id], form.frame()));
      },
      knn);
  return knn;
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1,
          typename RandomIt>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &aabb_metric,
                     const F1 &closest_point_f,
                     nearest_neighbors<RandomIt> &&knn) {
  nearness_search(form, aabb_metric, closest_point_f, knn);
  return knn;
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename F>
auto nearness_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1,
                     const F &closest_point_f) {
  return nearness_search(
      form0.tree(), form1.tree(),
      [&](const auto &aabb0, const auto &aabb1) {
        return tf::make_aabb_metrics(tf::transformed(aabb0, form0.frame()),
                                     tf::transformed(aabb1, form1.frame()));
      },
      [&](auto id0, auto id1) {
        return closest_point_f(tf::transformed(form0[id0], form0.frame()),
                               tf::transformed(form1[id1], form1.frame()));
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename F>
auto nearness_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1,
                     const F &closest_point_f,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  return nearness_search(
      form0.tree(), form1.tree(),
      [&](const auto &aabb0, const auto &aabb1) {
        return tf::make_aabb_metrics(tf::transformed(aabb0, form0.frame()),
                                     tf::transformed(aabb1, form1.frame()));
      },
      [&](auto id0, auto id1) {
        return closest_point_f(tf::transformed(form0[id0], form0.frame()),
                               tf::transformed(form1[id1], form1.frame()));
      },
      radius);
}
} // namespace tf
