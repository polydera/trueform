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
#include "./intersection.hpp"

namespace tf::intersect {

template <typename Index, typename Policy, typename Edges>
auto duplicate_intersection1(
    tf::intersect::intersection<Index> i,
    const tf::edge_membership_like<Policy> &em, const Edges &edges,
    tf::buffer<tf::intersect::intersection<Index>> &buffer) {
  auto push_f = [&](auto i) {
    buffer.push_back(i);
    std::swap(i.target, i.target_other);
    std::swap(i.object, i.object_other);
    buffer.push_back(i);
  };
  if (i.target_other.label == tf::topo_type::edge) {
    push_f(i);
  } else if (i.target_other.label == tf::topo_type::vertex) {
    auto vid = Index(edges[i.object_other][i.target_other.id]);
    for (auto edge_id1 : em[vid]) {
      i.object_other = edge_id1;
      i.target_other.id =
          Index(Index(edges[edge_id1][0]) == vid ? Index(0) : Index(1));
      push_f(i);
    }
  }
}

template <typename Index, typename Policy, typename Edges>
auto duplicate_intersection(
    tf::intersect::intersection<Index> i,
    const tf::edge_membership_like<Policy> &em, const Edges &edges,
    tf::buffer<tf::intersect::intersection<Index>> &buffer) {
  if (i.target.label == tf::topo_type::edge) {
    duplicate_intersection1(i, em, edges, buffer);
  } else if (i.target.label == tf::topo_type::vertex) {
    auto vid = Index(edges[i.object][i.target.id]);
    for (auto edge_id0 : em[vid]) {
      i.object = edge_id0;
      i.target.id =
          Index(Index(edges[edge_id0][0]) == vid ? Index(0) : Index(1));
      duplicate_intersection1(i, em, edges, buffer);
    }
  }
}

} // namespace tf::intersect
