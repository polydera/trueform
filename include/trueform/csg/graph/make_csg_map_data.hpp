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
#include "../../arrangement/construct/arrangement_map_data.hpp"
#include "../../arrangement/partition/partition_ids.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "tbb/task_group.h"

#include <algorithm>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Build a vertex-remap @ref tf::arrangement::arrangement_point_map_data
///        covering ALL labels of every form's
///        @ref tf::arrangement::partition_ids in one pass.
///
/// Each form's `partition_ids` carries **multiple labels** (here: `0` =
/// reverse-emit, `1` = forward-emit) and vertex discovery covers **every**
/// label: direction only matters at emission time.
///
/// Vertex discovery walks the arrangement's exposed triangle stream, so
/// every created point the triangulation materialized (intersection,
/// recovery split, refinement) enters the global vertex space.
///
/// Pass 1 (cut loops) is sequential across forms because the
/// `created_map` is shared globally. Pass 2 (uncut polygons) is
/// parallel per form.
template <typename Index, typename Arrangement, typename ApplyToPolygons>
auto make_csg_map_data(
    const Arrangement &arrangement,
    const tf::small_vector<tf::arrangement::partition_ids<Index>, 4> &pids,
    const ApplyToPolygons &apply_to_polygons)
    -> tf::arrangement::arrangement_point_map_data<Index> {
  const Index n_created_points =
      static_cast<Index>(arrangement.created_points().size());
  auto tris = arrangement.global().exposed_tris();
  auto tag_offsets = arrangement.global().tag_offsets();
  auto n_meshes = static_cast<Index>(pids.size());
  tf::arrangement::arrangement_point_map_data<Index> d;
  d.n_meshes = n_meshes;
  d.original_ids.resize(n_meshes);
  d.point_offsets.allocate(n_meshes + 1);
  d.point_offsets[0] = 0;
  for (Index i = 0; i < n_meshes; ++i)
    apply_to_polygons(i, [&](const auto &p) {
      d.original_ids[i].reserve(p.points().size());
      d.point_offsets[i + 1] = d.point_offsets[i] + p.points().size();
    });

  d.original_map.allocate(d.point_offsets.back());
  const Index sentinel_orig = static_cast<Index>(d.original_map.size());
  tf::parallel_fill(d.original_map, sentinel_orig);

  d.created_map.allocate(n_created_points);
  const Index sentinel_created = static_cast<Index>(d.created_map.size());
  tf::parallel_fill(d.created_map, sentinel_created);

  d.original_offsets.allocate(n_meshes + 1);
  std::fill(d.original_offsets.begin(), d.original_offsets.end(), 0);

  d.created_ids.reserve(n_created_points);
  Index create_current = 0;

  // Pass 1: cut-loop vertex discovery. Sequential across forms — the
  // created_map is shared globally, so created-vertex bookkeeping must
  // not race.
  for (Index t = 0; t < n_meshes; ++t) {
    auto &ids = d.original_ids[t];
    auto &curr = d.original_offsets[t + 1];
    const auto off = d.point_offsets[t];

    auto mark = [&](const auto &v) {
      if (v.source == tf::intersect::graph::vertex_source::original) {
        // The stream states originals FLAT.
        if (d.original_map[v.id] == sentinel_orig) {
          d.original_map[v.id] = curr++;
          ids.push_back(v.id - off);
        }
      } else {
        if (d.created_map[v.id] == sentinel_created) {
          d.created_map[v.id] = create_current++;
          d.created_ids.push_back(v.id);
        }
      }
    };
    const auto &cut_ids = pids[t].cut_faces;
    const Index n_labels = static_cast<Index>(cut_ids.size());
    for (Index label = 0; label < n_labels; ++label)
      for (auto lid : cut_ids[label])
        for (const auto &v : tris[tag_offsets[t] + lid])
          mark(v);
  }

  // Pass 2: uncut polygons. Parallel per form — only original
  // vertices are touched per form, no cross-form races.
  tbb::task_group tg;
  for (Index t = 0; t < n_meshes; ++t) {
    tg.run([&, t] {
      auto &ids = d.original_ids[t];
      auto &curr = d.original_offsets[t + 1];
      const auto off = d.point_offsets[t];

      apply_to_polygons(t, [&](const auto &polygons) {
        const auto &poly_ids = pids[t].polygons;
        const Index n_labels = static_cast<Index>(poly_ids.size());
        for (Index label = 0; label < n_labels; ++label) {
          for (auto face : tf::make_indirect_range(poly_ids[label],
                                                    polygons.faces())) {
            for (auto v : face) {
              auto flat = off + Index(v);
              if (d.original_map[flat] == sentinel_orig) {
                d.original_map[flat] = curr++;
                ids.push_back(Index(v));
              }
            }
          }
        }
      });
    });
  }
  tg.wait();

  for (Index i = 0; i < n_meshes; ++i)
    d.original_offsets[i + 1] += d.original_offsets[i];

  d.total_original_points = d.original_offsets.back();
  d.total_created_points = create_current;

  return d;
}

} // namespace tf::csg::graph
