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
#include "../../core/none.hpp"
#include "../../core/segments.hpp"
#include "../../spatial/aabb_tree.hpp"
#include "../../spatial/policy/tree.hpp"
#include "../../topology/edge_membership.hpp"
#include "../../topology/policy/edge_membership.hpp"
#include "tbb/parallel_invoke.h"
#include <tuple>

namespace tf::cut::dispatch {

template <typename Policy, typename... Ts>
auto tag_with_structures(const tf::segments<Policy> &s, std::tuple<Ts...> &t) {
  return std::apply(
      [&s](auto &...structs) { return (s | ... | tf::tag(structs)); }, t);
}

template <typename Policy>
auto make_missing_segment_structures(const tf::segments<Policy> &segments) {
  using Index = std::decay_t<decltype(segments.edges()[0][0])>;
  using CoordType = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;

  if constexpr (tf::has_tree_policy<Policy> &&
                tf::has_edge_membership_policy<Policy>) {
    return tf::none;
  } else if constexpr (!tf::has_tree_policy<Policy> &&
                       !tf::has_edge_membership_policy<Policy>) {
    tf::aabb_tree<Index, CoordType, Dims> tree;
    tf::edge_membership<Index> em;
    tbb::parallel_invoke(
        [&] { tree.build(segments, tf::config_tree(4, 4)); },
        [&] { em.build(segments); });
    return std::make_tuple(std::move(em), std::move(tree));
  } else if constexpr (!tf::has_tree_policy<Policy>) {
    tf::aabb_tree<Index, CoordType, Dims> tree;
    tree.build(segments, tf::config_tree(4, 4));
    return std::make_tuple(std::move(tree));
  } else {
    tf::edge_membership<Index> em;
    em.build(segments);
    return std::make_tuple(std::move(em));
  }
}

template <typename Policy, typename F>
auto segment_dispatch(const tf::segments<Policy> &segments, F &&f) {
  using S = decltype(make_missing_segment_structures(segments));

  if constexpr (std::is_same_v<S, tf::none_t>) {
    return f(segments);
  } else {
    auto s = make_missing_segment_structures(segments);
    return f(tag_with_structures(segments, s));
  }
}

} // namespace tf::cut::dispatch
