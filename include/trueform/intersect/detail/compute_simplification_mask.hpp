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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/range.hpp"
#include "../../core/tuple_hash.hpp"
#include "../types/intersection.hpp"
#include "../intersection_type.hpp"
#include "tbb/parallel_sort.h"

namespace tf::intersect {
template <typename Index, typename Policy>
auto compute_simplification_mask(
    tf::buffer<intersection<Index>> &intersection_ids,
    const tf::edges<Policy> &edges) {
  tbb::parallel_sort(intersection_ids, [](const auto &a0, const auto &a1) {
    return tf::make_intersection_type(a0.target.label, a0.target_other.label) <
           tf::make_intersection_type(a1.target.label, a1.target_other.label);
  });

  tf::hash_set<std::tuple<tf::intersection_type, Index, Index>,
               tf::tuple_hash<tf::intersection_type, Index, Index>>
      set;

  auto not_seen_yet =
      [&](tf::topo_id<Index> target,
          tf::topo_id<Index> target_other) {
        return set.find(std::make_tuple(
                   tf::make_intersection_type(target.label, target_other.label),
                   target.id, target_other.id)) == set.end();
      };

  set.reserve(intersection_ids.size());
  tf::buffer<char> id_mask;
  id_mask.allocate(intersection_ids.size());
  std::array<Index, 4> id_counts{};

  auto check_vertex_vertex_plain = [&](Index id0, Index id1) {
    if (id1 < id0)
      std::swap(id0, id1);
    return not_seen_yet({id0, tf::topo_type::vertex},
                        {id1, tf::topo_type::vertex});
  };

  auto check_vertex_edge = [&](Index id0, Index id1) {
    return !id_counts[0] || (check_vertex_vertex_plain(id0, edges[id1][0]) &&
                             check_vertex_vertex_plain(id0, edges[id1][1]));
  };

  auto check_edge_edge = [&](Index id0, Index id1) {
    return !id_counts[1] ||
           ((not_seen_yet({edges[id0][0], tf::topo_type::vertex},
                          {id1, tf::topo_type::edge}) &&
             not_seen_yet({edges[id0][1], tf::topo_type::vertex},
                          {id1, tf::topo_type::edge})) &&
            (not_seen_yet({edges[id1][0], tf::topo_type::vertex},
                          {id0, tf::topo_type::edge}) &&
             not_seen_yet({edges[id1][1], tf::topo_type::vertex},
                          {id0, tf::topo_type::edge})) &&
            check_vertex_edge(edges[id0][0], id1) &&
            check_vertex_edge(edges[id0][1], id1));
  };
  Index sequential_offset = 0;
  for (const intersection<Index> &e : intersection_ids) {
    auto type =
        tf::make_intersection_type(e.target.label, e.target_other.label);
    if (static_cast<int>(type) >=
        static_cast<int>(tf::intersection_type::edge_edge))
      break;
    id_counts[static_cast<int>(type)]++;
    sequential_offset++;
    switch (type) {
    case tf::intersection_type::vertex_vertex: {
      auto id0 = e.target.id;
      auto id1 = e.target_other.id;
      if (id1 < id0) // keep canonical order
        std::swap(id0, id1);
      set.insert({type, id0, id1});
      id_mask[e.id] = true;
      break;
    }
    case tf::intersection_type::vertex_edge: {
      if (check_vertex_edge(e.target.id, e.target_other.id)) {
        set.insert({type, e.target.id, e.target_other.id});
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
      // NOTE: we canonically keep only vertex-edge and no edge-vertex for
      // simplicity
    default:
      break; // should not happen
    }
  }
  tf::parallel_for_each(
      tf::make_range(intersection_ids.begin() + sequential_offset,
                     intersection_ids.end()),
      [&](const auto &e) {
        if (check_edge_edge(e.target.id, e.target_other.id)) {
          id_mask[e.id] = true;
        } else {
          id_mask[e.id] = false;
        }
      });
  return id_mask;
}

} // namespace tf::intersect
