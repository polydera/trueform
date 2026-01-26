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
#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/range.hpp"
#include "../../core/tuple_hash.hpp"
#include "../types/intersection.hpp"
#include "../types/intersection_id.hpp"
#include "tbb/parallel_sort.h"

namespace tf::intersect {
template <typename Index, typename Handle0, typename Handle1>
auto compute_simplification_mask(
    tf::buffer<tf::intersect::intersection_id<Index>> &intersection_ids,
    const Handle0 &handle0, const Handle1 &handle1) {
  tbb::parallel_sort(intersection_ids, [](const auto &a0, const auto &a1) {
    return a0.type < a1.type;
  });

  tf::hash_set<std::array<Index, 4>, tf::array_hash<Index, 4>> set;

  auto not_seen_yet = [&](Index i0, Index i1, Index i2, Index i3) {
    return set.find(std::array<Index, 4>{i0, i1, i2, i3}) == set.end();
  };

  set.reserve(intersection_ids.size());
  tf::buffer<char> id_mask;
  id_mask.allocate(intersection_ids.size());
  std::array<Index, 8> id_counts{};

  auto check_vertex_edge = [&not_seen_yet, &id_counts](auto v0, auto e0,
                                                       auto e1) {
    return !id_counts[0] ||
           (not_seen_yet(intersection_id<Index>::vertex_tag, v0,
                         intersection_id<Index>::vertex_tag, e0) &&
            not_seen_yet(intersection_id<Index>::vertex_tag, v0,
                         intersection_id<Index>::vertex_tag, e1));
  };
  auto check_edge_vertex = [&not_seen_yet, &id_counts](auto e0, auto e1,
                                                       auto v0) {
    return !id_counts[0] ||
           (not_seen_yet(intersection_id<Index>::vertex_tag, e0,
                         intersection_id<Index>::vertex_tag, v0) &&
            not_seen_yet(intersection_id<Index>::vertex_tag, e1,
                         intersection_id<Index>::vertex_tag, v0));
  };

  auto check_edge_edge = [&not_seen_yet, &check_edge_vertex,
                          &id_counts](Index e0, Index e1, Index e2, Index e3) {
    return (!id_counts[2] ||
            (not_seen_yet(e0, e1, intersection_id<Index>::vertex_tag, e2) &&
             not_seen_yet(e0, e1, intersection_id<Index>::vertex_tag, e3))) &&
           (!id_counts[1] ||
            (not_seen_yet(intersection_id<Index>::vertex_tag, e0, e2, e3) &&
             not_seen_yet(intersection_id<Index>::vertex_tag, e1, e2, e3))) &&
           check_edge_vertex(e0, e1, e2) && check_edge_vertex(e0, e1, e3);
  };

  auto check_vertex_polygon = [&not_seen_yet, &handle1, &id_counts](Index v,
                                                                    Index p) {
    const auto &face = handle1.faces()[p];
    Index size = face.size();
    Index prev = size - 1;
    for (Index next = 0; next < size; prev = next++) {
      auto e0 = face[prev];
      auto v0 = e0;
      auto e1 = face[next];
      if (e1 < e0)
        std::swap(e0, e1);
      if ((id_counts[1] &&
           !not_seen_yet(intersection_id<Index>::vertex_tag, v, e0, e1)) ||
          (id_counts[0] &&
           !not_seen_yet(intersection_id<Index>::vertex_tag, v,
                         intersection_id<Index>::vertex_tag, v0)))
        return false;
    }
    return true;
  };

  auto check_polygon_vertex = [&not_seen_yet, &handle0, &id_counts](Index p,
                                                                    Index v) {
    const auto &face = handle0.faces()[p];
    Index size = face.size();
    Index prev = size - 1;
    for (Index next = 0; next < size; prev = next++) {
      auto e0 = face[prev];
      auto v0 = e0;
      auto e1 = face[next];
      if (e1 < e0)
        std::swap(e0, e1);
      if ((id_counts[2] &&
           !not_seen_yet(e0, e1, intersection_id<Index>::vertex_tag, v)) ||
          (id_counts[0] &&
           !not_seen_yet(intersection_id<Index>::vertex_tag, v0,
                         intersection_id<Index>::vertex_tag, v)))
        return false;
    }
    return true;
  };

  auto check_edge_polygon = [&not_seen_yet, &handle1,
                             &id_counts](Index v0, Index v1, Index p) {
    if (id_counts[4] &&
        (!not_seen_yet(intersection_id<Index>::vertex_tag, v0,
                       intersection_id<Index>::polygon_tag, p) ||
         !not_seen_yet(intersection_id<Index>::vertex_tag, v1,
                       intersection_id<Index>::polygon_tag, p)))
      return false;
    const auto &face = handle1.faces()[p];
    Index size = face.size();
    Index prev = size - 1;
    for (Index next = 0; next < size; prev = next++) {
      auto e0 = face[prev];
      auto e1 = face[next];
      auto ev0 = e0;
      if (e1 < e0)
        std::swap(e0, e1);
      if ((id_counts[3] && !not_seen_yet(v0, v1, e0, e1)) ||
          (id_counts[2] &&
           !not_seen_yet(v0, v1, intersection_id<Index>::vertex_tag, ev0)) ||
          (id_counts[1] &&
           (!not_seen_yet(intersection_id<Index>::vertex_tag, v0, e0, e1) ||
            !not_seen_yet(intersection_id<Index>::vertex_tag, v1, e0, e1))) ||
          (id_counts[0] &&
           (!not_seen_yet(intersection_id<Index>::vertex_tag, v0,
                          intersection_id<Index>::vertex_tag, ev0) ||
            !not_seen_yet(intersection_id<Index>::vertex_tag, v1,
                          intersection_id<Index>::vertex_tag, ev0))))
        return false;
    }
    return true;
  };

  auto check_polygon_edge = [&not_seen_yet, &handle0,
                             &id_counts](Index p, Index v0, Index v1) {
    if (id_counts[5] &&
        (!not_seen_yet(intersection_id<Index>::polygon_tag, p,
                       intersection_id<Index>::vertex_tag, v0) ||
         !not_seen_yet(intersection_id<Index>::polygon_tag, p,
                       intersection_id<Index>::vertex_tag, v1)))
      return false;
    const auto &face = handle0.faces()[p];
    Index size = face.size();
    Index prev = size - 1;
    for (Index next = 0; next < size; prev = next++) {
      auto e0 = face[prev];
      auto e1 = face[next];
      auto ev0 = e0;
      if (e1 < e0)
        std::swap(e0, e1);
      if ((id_counts[3] && !not_seen_yet(e0, e1, v0, v1)) ||
          !not_seen_yet(e0, e1, intersection_id<Index>::vertex_tag, v0) ||
          (id_counts[2] &&
           (!not_seen_yet(e0, e1, intersection_id<Index>::vertex_tag, v0) ||
            !not_seen_yet(e0, e1, intersection_id<Index>::vertex_tag, v1))) ||
          (id_counts[1] &&
           !not_seen_yet(intersection_id<Index>::vertex_tag, ev0, v0, v1)) ||
          (id_counts[0] &&
           (!not_seen_yet(intersection_id<Index>::vertex_tag, ev0,
                          intersection_id<Index>::vertex_tag, v0) ||
            !not_seen_yet(intersection_id<Index>::vertex_tag, ev0,
                          intersection_id<Index>::vertex_tag, v1))))
        return false;
    }
    return true;
  };

  Index sequential_offset = 0;
  for (const intersection_id<Index> &e : intersection_ids) {
    if (static_cast<int>(e.type) >= 6)
      break;
    id_counts[static_cast<int>(e.type)]++;
    sequential_offset++;
    switch (e.type) {
    case intersection_type::vertex_vertex: {
      set.insert({e.self_id0, e.self_id1, e.other_id0, e.other_id1});
      id_mask[e.id] = true;
      break;
    }
    case intersection_type::vertex_edge: {
      if (check_vertex_edge(e.self_id1, e.other_id0, e.other_id1)) {
        set.insert({e.self_id0, e.self_id1, e.other_id0, e.other_id1});
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    case intersection_type::edge_vertex: {
      if (check_edge_vertex(e.self_id0, e.self_id1, e.other_id1)) {
        set.insert({e.self_id0, e.self_id1, e.other_id0, e.other_id1});
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    case intersection_type::edge_edge: {
      if (check_edge_edge(e.self_id0, e.self_id1, e.other_id0, e.other_id1)) {
        set.insert({e.self_id0, e.self_id1, e.other_id0, e.other_id1});
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    case intersection_type::vertex_face: {
      if (check_vertex_polygon(e.self_id1, e.other_id1)) {
        set.insert({e.self_id0, e.self_id1, e.other_id0, e.other_id1});
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    case intersection_type::face_vertex: {
      if (check_polygon_vertex(e.self_id1, e.other_id1)) {
        set.insert({e.self_id0, e.self_id1, e.other_id0, e.other_id1});
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    default:
      break;
    }
  }
  tf::parallel_for_each(
      tf::make_range(intersection_ids.begin() + sequential_offset,
                     intersection_ids.end()),
      [&](const auto &e) {
        if (e.type == intersection_type::edge_face) {
          if (check_edge_polygon(e.self_id0, e.self_id1, e.other_id1)) {
            id_mask[e.id] = true;
          } else {
            id_mask[e.id] = false;
          }
        } else if (e.type == intersection_type::face_edge) {
          if (check_polygon_edge(e.self_id1, e.other_id0, e.other_id1)) {
            id_mask[e.id] = true;
          } else {
            id_mask[e.id] = false;
          }
        }
      });
  return id_mask;
}

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
      [&](tf::intersect::intersection_target<Index> target,
          tf::intersect::intersection_target<Index> target_other) {
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
