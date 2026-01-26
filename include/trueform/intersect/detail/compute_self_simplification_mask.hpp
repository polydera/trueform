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
#include "../../core/hash_set.hpp"
#include "../../core/range.hpp"
#include "../types/intersection_id.hpp"
#include "tbb/parallel_sort.h"

namespace tf::intersect {
template <typename Index, typename Handle>
auto compute_self_simplification_mask(
    tf::buffer<tf::intersect::intersection_id<Index>> &intersection_ids,
    const Handle &handle) {
  tbb::parallel_sort(intersection_ids, [](const auto &a0, const auto &a1) {
    return a0.type < a1.type;
  });

  tf::hash_set<std::array<Index, 4>, tf::array_hash<Index, 4>> set;

  auto not_seen_yet = [&](Index i0, Index i1, Index i2, Index i3) {
    auto k0 = std::make_pair(i0, i1);
    auto k1 = std::make_pair(i2, i3);
    if (k0 < k1)
      std::swap(k0, k1);
    return set.find(std::array<Index, 4>{k0.first, k0.second, k1.first,
                                         k1.second}) == set.end();
  };

  auto set_insert = [&](Index i0, Index i1, Index i2, Index i3) {
    auto k0 = std::make_pair(i0, i1);
    auto k1 = std::make_pair(i2, i3);
    if (k0 < k1)
      std::swap(k0, k1);
    set.insert(std::array<Index, 4>{k0.first, k0.second, k1.first, k1.second});
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

  auto check_vertex_polygon = [&not_seen_yet, &handle, &id_counts](Index v,
                                                                   Index p) {
    const auto &face = handle.faces()[p];
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

  auto check_polygon_vertex = [&not_seen_yet, &handle, &id_counts](Index p,
                                                                   Index v) {
    const auto &face = handle.faces()[p];
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

  auto check_edge_polygon = [&not_seen_yet, &handle,
                             &id_counts](Index v0, Index v1, Index p) {
    if (id_counts[4] &&
        (!not_seen_yet(intersection_id<Index>::vertex_tag, v0,
                       intersection_id<Index>::polygon_tag, p) ||
         !not_seen_yet(intersection_id<Index>::vertex_tag, v1,
                       intersection_id<Index>::polygon_tag, p)))
      return false;
    const auto &face = handle.faces()[p];
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

  auto check_polygon_edge = [&not_seen_yet, &handle,
                             &id_counts](Index p, Index v0, Index v1) {
    if (id_counts[5] &&
        (!not_seen_yet(intersection_id<Index>::polygon_tag, p,
                       intersection_id<Index>::vertex_tag, v0) ||
         !not_seen_yet(intersection_id<Index>::polygon_tag, p,
                       intersection_id<Index>::vertex_tag, v1)))
      return false;
    const auto &face = handle.faces()[p];
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
      set_insert(e.self_id0, e.self_id1, e.other_id0, e.other_id1);
      id_mask[e.id] = true;
      break;
    }
    case intersection_type::vertex_edge: {
      if (check_vertex_edge(e.self_id1, e.other_id0, e.other_id1)) {
        set_insert(e.self_id0, e.self_id1, e.other_id0, e.other_id1);
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    case intersection_type::edge_vertex: {
      if (check_edge_vertex(e.self_id0, e.self_id1, e.other_id1)) {
        set_insert(e.self_id0, e.self_id1, e.other_id0, e.other_id1);
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    case intersection_type::edge_edge: {
      if (check_edge_edge(e.self_id0, e.self_id1, e.other_id0, e.other_id1)) {
        set_insert(e.self_id0, e.self_id1, e.other_id0, e.other_id1);
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    case intersection_type::vertex_face: {
      if (check_vertex_polygon(e.self_id1, e.other_id1)) {
        set_insert(e.self_id0, e.self_id1, e.other_id0, e.other_id1);
        id_mask[e.id] = true;
      } else {
        id_mask[e.id] = false;
      }
      break;
    }
    case intersection_type::face_vertex: {
      if (check_polygon_vertex(e.self_id1, e.other_id1)) {
        set_insert(e.self_id0, e.self_id1, e.other_id0, e.other_id1);
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
} // namespace tf::intersect
