/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/make_equivalence_class_index_map.hpp"
#include "../../core/algorithm/make_unique_index_map.hpp"
#include "../../core/distance.hpp"
#include "../../core/index_map.hpp"
#include "../../core/points.hpp"
#include "../../spatial/gather_self_ids.hpp"
#include "../../spatial/policy/tree.hpp"

namespace tf {
template <typename Policy, typename Index>
auto make_clean_index_map(const tf::points<Policy> &points,
                          tf::index_map_buffer<Index> &im) {
  if (!points.size())
    return;
  tf::make_unique_index_map(points, im);
}

template <typename Policy, typename Index>
auto make_clean_index_map(const tf::points<Policy> &points,
                          tf::coordinate_type<Policy> tolerance,
                          tf::index_map_buffer<Index> &im) {
  if (!points.size())
    return;
  if (tolerance == 0)
    return make_clean_index_map(points, im);
  if constexpr (tf::has_tree_policy<Policy>) {
    tf::buffer<std::array<Index, 2>> ids;
    tf::gather_self_ids(
        tf::make_form(points),
        [d2 = tolerance * tolerance](const auto &x, const auto &y) {
          return distance2(x, y) <= d2;
        },
        std::back_inserter(ids));
    tf::make_dense_equivalence_class_index_map(ids, points.size(), im);
  } else {
    constexpr std::size_t dims = tf::static_size_v<decltype(points[0])>;
    tf::tree<Index, tf::coordinate_type<Policy>, dims> tree;
    tree.build(points, tf::config_tree(4, 4));
    return make_clean_index_map(points | tf::tag(tree), tolerance, im);
  }
}
} // namespace tf
