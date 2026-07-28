/*
 * Copyright (c) 2026 XLAB
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
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include <algorithm>
#include <array>

namespace tf::cut {

template <typename Index, typename Merge>
auto resolve_region_vertex_key(
    Index n_tags, Index tag,
    const tf::intersect::graph::vertex<Index> &vertex,
    const tf::buffer<Merge> &merges) -> std::array<Index, 2> {
  auto key = tf::cut::detail::region_vertex_key(n_tags, tag, vertex);
  if (merges.size() == 0)
    return key;
  auto merge = std::lower_bound(
      merges.begin(), merges.end(), key,
      [](const Merge &candidate, const std::array<Index, 2> &value) {
        return candidate.from < value;
      });
  return merge != merges.end() && merge->from == key ? merge->to_key : key;
}

template <typename Index, typename Merge>
auto resolve_region_vertex(
    Index n_tags, Index tag,
    const tf::intersect::graph::vertex<Index> &vertex,
    const tf::buffer<Merge> &merges)
    -> tf::intersect::graph::vertex<Index> {
  if (merges.size() == 0)
    return vertex;
  auto key = tf::cut::detail::region_vertex_key(n_tags, tag, vertex);
  auto merge = std::lower_bound(
      merges.begin(), merges.end(), key,
      [](const Merge &candidate, const std::array<Index, 2> &value) {
        return candidate.from < value;
      });
  if (merge == merges.end() || merge->from != key)
    return vertex;
  if (tf::cut::detail::region_key_is_created(n_tags, merge->to_key) ||
      merge->to_key[0] == tag)
    return merge->to;

  auto local = key;
  for (const auto &candidate : merges) {
    if (candidate.to_key != merge->to_key)
      continue;
    if ((tf::cut::detail::region_key_is_created(n_tags, candidate.from) ||
         candidate.from[0] == tag) &&
        candidate.from < local)
      local = candidate.from;
  }
  if (local == key)
    return vertex;
  return tf::intersect::graph::vertex<Index>{
      tf::cut::detail::region_key_is_created(n_tags, local)
          ? tf::intersect::graph::vertex_source::created
          : tf::intersect::graph::vertex_source::original,
      local[1],
      {0, tf::topo_type::face}};
}

} // namespace tf::cut
