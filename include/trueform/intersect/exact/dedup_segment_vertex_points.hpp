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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/make_equivalence_class_map.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/views/indirect_range.hpp"
#include "./intersection.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>

namespace tf::intersect {

/// Deduplicate vertex intersection points for segments.
/// Multiple records referencing the same vertex ID (from different
/// segment pairs) get merged to one point ID. VV records merge
/// cross-vertex pairs via union-find.
///
/// @param intersections Raw intersection records (target.id is local
///        index 0/1 for vertex targets).
/// @param points Raw int32 intersection points.
/// @param edges_lookup edges_lookup(object, local_id) → global vertex ID.
template <typename Index, std::size_t Dims, typename EdgesLookup>
void dedup_segment_vertex_points(
    tf::buffer<tf::intersect::intersection<Index>> &intersections,
    tf::buffer<tf::point<int32_t, Dims>> &points,
    const EdgesLookup &edges_lookup) {
  struct merge_pair {
    Index a, b;
  };
  struct idx_pt {
    Index idx, pt_id;
  };

  // Step 1: Collect all global vertex IDs from vertex targets
  tf::buffer<Index> vertex_ids;
  tf::generic_generate(
      tf::make_range(intersections), vertex_ids,
      [&](const tf::intersect::intersection<Index> &rec,
          tf::buffer<Index> &buf) {
        if (rec.target.label == tf::topo_type::vertex)
          buf.push_back(edges_lookup(rec.object, rec.target.id));
        if (rec.target_other.label == tf::topo_type::vertex)
          buf.push_back(edges_lookup(rec.object_other, rec.target_other.id));
      });

  if (vertex_ids.size() == 0)
    return;

  // Step 2: Sort + unique → K unique vertex IDs
  tbb::parallel_sort(vertex_ids.begin(), vertex_ids.end());
  auto uniq_end = std::unique(vertex_ids.begin(), vertex_ids.end());
  auto K = static_cast<std::size_t>(uniq_end - vertex_ids.begin());
  vertex_ids.reallocate(K);

  auto find_idx = [&](Index vid) -> Index {
    return Index(
        std::lower_bound(vertex_ids.begin(), vertex_ids.begin() + K, vid) -
        vertex_ids.begin());
  };

  // Step 3: Generate VV merge pairs + (dense_idx, pt_id) for vertex targets
  tf::buffer<merge_pair> pairs;
  tf::buffer<idx_pt> idx_pts;
  tf::generic_generate(
      tf::make_range(intersections), std::tie(pairs, idx_pts),
      [&](const tf::intersect::intersection<Index> &rec, auto &bufs) {
        auto &[pair_buf, ip_buf] = bufs;
        if (rec.target.label == tf::topo_type::vertex) {
          auto a = find_idx(edges_lookup(rec.object, rec.target.id));
          ip_buf.push_back({a, rec.id});
          if (rec.target_other.label == tf::topo_type::vertex) {
            auto b = find_idx(
                edges_lookup(rec.object_other, rec.target_other.id));
            ip_buf.push_back({b, rec.id});
            if (a != b)
              pair_buf.push_back({a, b});
          }
        } else if (rec.target_other.label == tf::topo_type::vertex) {
          ip_buf.push_back(
              {find_idx(edges_lookup(rec.object_other, rec.target_other.id)),
               rec.id});
        }
      });

  // Step 4: Union-find → equivalence classes
  tf::buffer<Index> eq_map;
  eq_map.allocate(K);
  auto n_classes = tf::make_dense_equivalence_class_map(pairs, eq_map);

  // Step 5: Build point_map[old_id] → new_id
  const auto none = static_cast<Index>(-1);
  tf::buffer<Index> point_map;
  point_map.allocate(points.size());
  tf::parallel_fill(point_map, none);

  // Vertex points: all old IDs for the same class → one new ID
  for (const auto &e : idx_pts)
    point_map[e.pt_id] = eq_map[e.idx];

  // Non-vertex points (EE): sequential IDs after vertex classes
  Index next_id = Index(n_classes);
  for (auto &e : point_map)
    if (e == none)
      e = next_id++;

  // Step 6: Build compact points + remap records
  tf::buffer<tf::point<int32_t, Dims>> new_pts;
  new_pts.allocate(next_id);

  // Parallel copy (idempotent: same-class vertex writes identical data)
  tf::parallel_copy(points, tf::make_indirect_range(point_map, new_pts));

  tf::parallel_for_each(
      intersections, [&](tf::intersect::intersection<Index> &rec) {
        rec.id = point_map[rec.id];
      });

  points = std::move(new_pts);
}

} // namespace tf::intersect
