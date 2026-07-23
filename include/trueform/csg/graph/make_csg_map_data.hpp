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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../cut/construct/arrangement_map_data.hpp"
#include "../../cut/region_triangulator.hpp"
#include "../../cut/partition/partition_ids.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "tbb/task_group.h"

#include <algorithm>
#include <array>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Build a vertex-remap @ref tf::cut::partition_map_data
///        covering ALL labels of every form's
///        @ref tf::cut::partition_ids in one pass.
///
/// Variant of @ref tf::cut::make_partition_map_data tailored for the
/// CSG output case, where each form's `partition_ids` carries
/// **multiple labels** (here: `0` = reverse-emit, `1` = forward-emit)
/// and we need vertex discovery to cover **every** label (direction
/// only matters at emission time, not at vertex discovery).
///
/// Vertex discovery walks the graph's exposed triangle-grain loops, so
/// every created point the triangulation materialized (intersection,
/// recovery split, refinement) enters the global vertex space.
/// `n_created_points` is the size of the graph's unified created-points
/// buffer.
///
/// Pass 1 (cut loops) is sequential across forms because the
/// `created_map` is shared globally. Pass 2 (uncut polygons) is
/// parallel per form.
template <typename Index, typename Int, typename ApplyToPolygons>
auto make_csg_map_data(const tf::cut::region_triangulator<Index, Int> &rt,
                       Index n_created_points,
                       const tf::small_vector<tf::cut::partition_ids<Index>, 4>
                           &pids,
                       const ApplyToPolygons &apply_to_polygons)
    -> tf::cut::partition_map_data<Index> {
  auto n_meshes = static_cast<Index>(pids.size());
  tf::cut::partition_map_data<Index> d;
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
        auto flat = off + v.id;
        if (d.original_map[flat] == sentinel_orig) {
          d.original_map[flat] = curr++;
          ids.push_back(v.id);
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
    auto loops = rt.loops();
    for (Index label = 0; label < n_labels; ++label) {
      for (auto lid : cut_ids[label]) {
        const Index gli = rt.tag_offsets()[t] + lid;
        for (const auto &v : loops[gli])
          mark(v);
      }
    }
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

  if (rt.merges().size() != 0) {
    tf::parallel_fill(d.original_map, Index(0));
    tf::parallel_fill(d.created_map, Index(0));

    auto mark_key = [&](const std::array<Index, 2> &key) {
      if (key[0] != rt.n_tags())
        d.original_map[d.point_offsets[key[0]] + key[1]] = Index(1);
      else
        d.created_map[key[1]] = Index(1);
    };

    using vertex_t = tf::intersect::graph::vertex<Index>;
    using source_t = tf::intersect::graph::vertex_source;
    for (Index t = 0; t < n_meshes; ++t)
      for (Index id : d.original_ids[t])
        mark_key(rt.resolve_key(
            t, vertex_t{source_t::original, id, {0, tf::topo_type::face}}));
    for (Index id : d.created_ids)
      mark_key(rt.resolve_key(
          Index(0),
          vertex_t{source_t::created, id, {0, tf::topo_type::face}}));

    std::fill(d.original_offsets.begin(), d.original_offsets.end(), Index(0));
    for (Index t = 0; t < n_meshes; ++t) {
      auto &ids = d.original_ids[t];
      ids.clear();
      Index current = 0;
      const Index begin = d.point_offsets[t];
      const Index end = d.point_offsets[t + 1];
      for (Index flat = begin; flat < end; ++flat) {
        if (d.original_map[flat]) {
          d.original_map[flat] = current++;
          ids.push_back(flat - begin);
        } else {
          d.original_map[flat] = sentinel_orig;
        }
      }
      d.original_offsets[t + 1] = current;
    }

    d.created_ids.clear();
    create_current = 0;
    for (Index id = 0; id < n_created_points; ++id) {
      if (d.created_map[id]) {
        d.created_map[id] = create_current++;
        d.created_ids.push_back(id);
      } else {
        d.created_map[id] = sentinel_created;
      }
    }
  }

  for (Index i = 0; i < n_meshes; ++i)
    d.original_offsets[i + 1] += d.original_offsets[i];

  d.total_original_points = d.original_offsets.back();
  d.total_created_points = create_current;

  return d;
}

} // namespace tf::csg::graph
