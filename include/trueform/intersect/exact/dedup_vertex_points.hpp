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
#include "../../core/views/indirect_range.hpp"
#include "../../exact/vertex.hpp"
#include "./tagged_intersection.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>

namespace tf::intersect {

/// Deduplicate vertex intersection points. Multiple records for the same
/// mesh vertex (from different face pairs) get merged to one point ID.
/// VV records also merge cross-mesh vertex pairs via union-find.
template <typename Index, typename Int, typename FacesLookup>
void dedup_vertex_points(tf::buffer<tagged_intersection<Index>> &intersections,
                         tf::buffer<tf::exact::pt3<Int>> &points,
                         const FacesLookup &faces) {
  struct tag_vid {
    Index tag, vid;
    auto operator<(const tag_vid &o) const -> bool {
      if (tag != o.tag)
        return tag < o.tag;
      return vid < o.vid;
    }
    auto operator==(const tag_vid &o) const -> bool {
      return tag == o.tag && vid == o.vid;
    }
  };
  struct merge_pair {
    Index a, b;
  };
  struct idx_pt {
    Index idx, pt_id;
  };

  // Step 1: Collect all (tag, global_vid) from vertex targets
  tf::buffer<tag_vid> entries;
  tf::generic_generate(
      tf::make_range(intersections), entries,
      [&](const tagged_intersection<Index> &rec, tf::buffer<tag_vid> &buf) {
        if (rec.target.label == tf::topo_type::vertex)
          buf.push_back({rec.tag, faces(rec.tag, rec.object, rec.target.id)});
        if (rec.target_other.label == tf::topo_type::vertex)
          buf.push_back({rec.tag_other, faces(rec.tag_other, rec.object_other,
                                              rec.target_other.id)});
      });

  if (entries.size() == 0)
    return;

  // Step 2: Sort + unique → K unique (tag, vid) entries
  tbb::parallel_sort(entries.begin(), entries.end());
  auto uniq_end = std::unique(entries.begin(), entries.end());
  auto K = static_cast<std::size_t>(uniq_end - entries.begin());
  entries.reallocate(K);

  auto find_idx = [&](Index tag, Index vid) -> Index {
    tag_vid key{tag, vid};
    return Index(std::lower_bound(entries.begin(), entries.end(), key) -
                 entries.begin());
  };

  // Step 3: Generate VV merge pairs + (dense_idx, pt_id) for all vertex
  // targets. Uses tuple-of-buffers overload for single parallel pass.
  tf::buffer<merge_pair> pairs;
  tf::buffer<idx_pt> idx_pts;
  tf::generic_generate(
      tf::make_range(intersections), std::tie(pairs, idx_pts),
      [&](const tagged_intersection<Index> &rec, auto &bufs) {
        auto &[pair_buf, ip_buf] = bufs;
        if (rec.target.label == tf::topo_type::vertex) {
          auto a = find_idx(rec.tag, faces(rec.tag, rec.object, rec.target.id));
          ip_buf.push_back({a, rec.id});
          if (rec.target_other.label == tf::topo_type::vertex) {
            auto b =
                find_idx(rec.tag_other, faces(rec.tag_other, rec.object_other,
                                              rec.target_other.id));
            ip_buf.push_back({b, rec.id});
            if (a != b)
              pair_buf.push_back({a, b});
          }
        } else if (rec.target_other.label == tf::topo_type::vertex) {
          ip_buf.push_back(
              {find_idx(rec.tag_other, faces(rec.tag_other, rec.object_other,
                                             rec.target_other.id)),
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

  // Vertex points: map all old IDs for the same class to one new ID
  for (const auto &e : idx_pts)
    point_map[e.pt_id] = eq_map[e.idx];

  // Non-vertex points: assign new IDs starting after vertex classes
  Index next_id = Index(n_classes);
  for (auto &e : point_map)
    if (e == none)
      e = next_id++;

  // Step 6: Build compact points + remap records
  tf::buffer<tf::exact::pt3<Int>> new_pts;
  new_pts.allocate(next_id);

  // Parallel copy (idempotent: same-class vertex writes identical data)
  tf::parallel_copy(points, tf::make_indirect_range(point_map, new_pts));
  //
  tf::parallel_for_each(intersections, [&](tagged_intersection<Index> &rec) {
    rec.id = point_map[rec.id];
  });

  points = std::move(new_pts);
}

} // namespace tf::intersect
