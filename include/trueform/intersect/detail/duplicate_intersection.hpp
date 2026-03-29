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
#include "../../core/buffer.hpp"
#include "../../topology/edge_membership_like.hpp"
#include "../types/intersection.hpp"

namespace tf::intersect {

namespace detail {
template <typename Index, typename Policy>
auto duplicate_intersection1(
    intersect::intersection<Index> i,
    const tf::edge_membership_like<Policy> &em,
    tf::buffer<intersect::intersection<Index>> &buffer) {
  auto push_f = [&](auto i) {
    buffer.push_back(i);
    std::swap(i.target, i.target_other);
    std::swap(i.object, i.object_other);
    buffer.push_back(i);
  };
  if (i.target_other.label == tf::topo_type::edge) {
    push_f(i);
  } else if (i.target_other.label == tf::topo_type::vertex) {
    for (auto edge_id1 : em[i.target_other.id]) {
      i.object_other = edge_id1;
      push_f(i);
    }
  }
}
} // namespace detail
template <typename Index, typename Policy>
auto duplicate_intersection(
    intersect::intersection<Index> i,
    const tf::edge_membership_like<Policy> &em,
    tf::buffer<intersect::intersection<Index>> &buffer) {
  if (i.target.label == tf::topo_type::edge) {
    detail::duplicate_intersection1(i, em, buffer);
  } else if (i.target.label == tf::topo_type::vertex) {
    for (auto edge_id0 : em[i.target.id]) {
      i.object = edge_id0;
      detail::duplicate_intersection1(i, em, buffer);
    }
  }
}

} // namespace tf::intersect
