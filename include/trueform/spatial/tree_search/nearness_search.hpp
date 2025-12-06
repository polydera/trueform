/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../../core/coordinate_type.hpp"
#include "../../core/make_buffer_for_transformed.hpp"
#include "../../core/make_local_buffer_for_transformed.hpp"
#include "../../core/policy/buffer.hpp"
#include "../../core/transformed.hpp"
#include "../form.hpp"
#include "../nearest_neighbors.hpp"
#include "../tree/dual_proximity.hpp"
#include "../tree/local_tree_metric_result.hpp"
#include "../tree/proximity.hpp"
#include "../tree/traversal_metrics.hpp"
#include "../tree/tree_metric_result.hpp"
#include "../tree_like.hpp"
#include "../tree_metric_info.hpp"
#include "../tree_metric_info_pair.hpp"

namespace tf::spatial {

// ============================================================================
// Single-tree overloads
// ============================================================================

template <typename TreePolicy, typename F0, typename F1>
auto nearness_search(const tf::tree_like<TreePolicy> &tree, const F0 &bv_metric,
                     const F1 &closest_point_f) {
  using Index = typename TreePolicy::index_type;
  using real_t = typename TreePolicy::coordinate_type;

  using tree_metric_t =
      tf::tree_metric_info<Index, decltype(closest_point_f(Index(0)))>;
  tree_metric_result<tree_metric_t> result{std::numeric_limits<real_t>::max()};
  impl::proximity(tree, bv_metric, closest_point_f, result);
  return result.info;
}

template <typename TreePolicy, typename F0, typename F1>
auto nearness_search(const tf::tree_like<TreePolicy> &tree, const F0 &bv_metric,
                     const F1 &closest_point_f,
                     typename TreePolicy::coordinate_type radius) {
  using Index = typename TreePolicy::index_type;

  using tree_metric_t =
      tf::tree_metric_info<Index, decltype(closest_point_f(Index(0)))>;
  tree_metric_result<tree_metric_t> result{radius * radius};
  impl::proximity(tree, bv_metric, closest_point_f, result);
  return result.info;
}

template <typename TreePolicy, typename F0, typename F1, typename RandomIt>
auto nearness_search(const tf::tree_like<TreePolicy> &tree, const F0 &bv_metric,
                     const F1 &closest_point_f,
                     tf::nearest_neighbors<RandomIt> &knn)
    -> tf::nearest_neighbors<RandomIt> & {
  impl::proximity(tree, bv_metric, closest_point_f, knn);
  return knn;
}

template <typename TreePolicy, typename F0, typename F1, typename RandomIt>
auto nearness_search(const tf::tree_like<TreePolicy> &tree, const F0 &bv_metric,
                     const F1 &closest_point_f,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  impl::proximity(tree, bv_metric, closest_point_f, knn);
  return knn;
}

// ============================================================================
// Dual-tree overloads
// ============================================================================

template <typename TreePolicy0, typename TreePolicy1, typename F0, typename F1>
auto nearness_search(const tf::tree_like<TreePolicy0> &tree0,
                     const tf::tree_like<TreePolicy1> &tree1,
                     const F0 &bv_metrics_f, const F1 &closest_points_f) {
  using Index0 = typename TreePolicy0::index_type;
  using Index1 = typename TreePolicy1::index_type;
  using real_t = typename TreePolicy0::coordinate_type;

  using tree_metric_t = tf::tree_metric_info_pair<
      Index0, Index1, decltype(closest_points_f(Index0(0), Index1(0)))>;
  local_tree_metric_result<tree_metric_t> result{
      std::numeric_limits<real_t>::max()};
  impl::dual_proximity(tree0, tree1, bv_metrics_f, closest_points_f, result);
  return result.info();
}

template <typename TreePolicy0, typename TreePolicy1, typename F0, typename F1>
auto nearness_search(const tf::tree_like<TreePolicy0> &tree0,
                     const tf::tree_like<TreePolicy1> &tree1,
                     const F0 &bv_metrics_f, const F1 &closest_points_f,
                     typename TreePolicy0::coordinate_type radius) {
  using Index0 = typename TreePolicy0::index_type;
  using Index1 = typename TreePolicy1::index_type;

  using tree_metric_t = tf::tree_metric_info_pair<
      Index0, Index1, decltype(closest_points_f(Index0(0), Index1(0)))>;
  local_tree_metric_result<tree_metric_t> result{radius * radius};
  impl::dual_proximity(tree0, tree1, bv_metrics_f, closest_points_f, result);
  return result.info();
}

// ============================================================================
// Form overloads (single-tree)
// ============================================================================

template <std::size_t Dims, typename Policy0, typename F0, typename F1>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &bv_metric,
                     const F1 &closest_point_f) {
  auto buff = tf::core::make_buffer_for_transformed(form[0], form.frame());
  return nearness_search(
      form.tree(),
      [&](const auto &bv) {
        return bv_metric(tf::transformed(bv, form.frame()));
      },
      [&](auto id) {
        return closest_point_f(
            tf::transformed(form[id] | tf::tag(buff), form.frame()));
      });
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &bv_metric,
                     const F1 &closest_point_f,
                     tf::coordinate_type<Policy0> radius) {
  auto buff = tf::core::make_buffer_for_transformed(form[0], form.frame());
  return nearness_search(
      form.tree(),
      [&](const auto &bv) {
        return bv_metric(tf::transformed(bv, form.frame()));
      },
      [&](auto id) {
        return closest_point_f(
            tf::transformed(form[id] | tf::tag(buff), form.frame()));
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1,
          typename RandomIt>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &bv_metric,
                     const F1 &closest_point_f,
                     tf::nearest_neighbors<RandomIt> &knn)
    -> tf::nearest_neighbors<RandomIt> & {
  auto buff = tf::core::make_buffer_for_transformed(form[0], form.frame());
  nearness_search(
      form.tree(),
      [&](const auto &bv) {
        return bv_metric(tf::transformed(bv, form.frame()));
      },
      [&](auto id) {
        return closest_point_f(
            tf::transformed(form[id] | tf::tag(buff), form.frame()));
      },
      knn);
  return knn;
}

template <std::size_t Dims, typename Policy0, typename F0, typename F1,
          typename RandomIt>
auto nearness_search(const tf::form<Dims, Policy0> &form, const F0 &bv_metric,
                     const F1 &closest_point_f,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  nearness_search(form, bv_metric, closest_point_f, knn);
  return knn;
}

// ============================================================================
// Form overloads (dual-tree)
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1, typename F>
auto nearness_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1,
                     const F &closest_point_f) {
  auto buff0 =
      tf::core::make_local_buffer_for_transformed(form0[0], form0.frame());
  auto buff1 =
      tf::core::make_local_buffer_for_transformed(form1[0], form1.frame());
  return nearness_search(
      form0.tree(), form1.tree(),
      [&](const auto &bv0, const auto &bv1) {
        return traversal_metrics(tf::transformed(bv0, form0.frame()),
                                 tf::transformed(bv1, form1.frame()));
      },
      [&](auto id0, auto id1) {
        return closest_point_f(
            tf::transformed(form0[id0] | tf::tag(buff0), form0.frame()),
            tf::transformed(form1[id1] | tf::tag(buff1), form1.frame()));
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename F>
auto nearness_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1,
                     const F &closest_point_f,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  auto buff0 =
      tf::core::make_local_buffer_for_transformed(form0[0], form0.frame());
  auto buff1 =
      tf::core::make_local_buffer_for_transformed(form1[0], form1.frame());
  return nearness_search(
      form0.tree(), form1.tree(),
      [&](const auto &bv0, const auto &bv1) {
        return traversal_metrics(tf::transformed(bv0, form0.frame()),
                                 tf::transformed(bv1, form1.frame()));
      },
      [&](auto id0, auto id1) {
        return closest_point_f(
            tf::transformed(form0[id0] | tf::tag(buff0), form0.frame()),
            tf::transformed(form1[id1] | tf::tag(buff1), form1.frame()));
      },
      radius);
}

} // namespace tf::spatial
