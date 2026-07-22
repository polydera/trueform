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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/zip.hpp"
#include "../partition/partition_ids.hpp"
#include "./arrangement_map_data.hpp"
#include "tbb/task_group.h"

namespace tf::cut {

/// Created-id discovery shared by the map-data builders: only ids the
/// exposure references get output slots (benign concurrent marks).
template <typename Index, typename Cuts, typename D>
auto discover_created(const Cuts &fc, Index n_created_points, D &d) -> void {
  d.created_map.allocate(std::size_t(n_created_points));
  tf::parallel_fill(d.created_map, Index(0));
  tf::parallel_for_each(
      fc.loops(),
      [&](const auto &loop) {
        for (const auto &v : loop)
          if (v.source == tf::intersect::graph::vertex_source::created)
            d.created_map[std::size_t(v.id)] = Index(1);
      },
      tf::checked);
  d.created_ids.reserve(std::size_t(n_created_points));
  Index create_current = 0;
  for (Index i = 0; i < n_created_points; ++i)
    if (d.created_map[std::size_t(i)]) {
      d.created_map[std::size_t(i)] = create_current++;
      d.created_ids.push_back(i);
    } else {
      d.created_map[std::size_t(i)] = n_created_points;
    }
  d.total_created_used = create_current;
}

/// Build arrangement map data: vertex ID maps, face masks, offset tables.
///
/// Walks cut face loops and uncut faces in parallel per mesh, assigns
/// contiguous IDs to original vertices on first encounter. Created
/// vertices (intersection points) are all used — offset after originals.
template <typename Index, typename Cuts, typename ApplyToPolygons>
auto make_arrangement_map_data(const Cuts &fc,
                               const ApplyToPolygons &apply_to_polygons,
                               Index n_meshes, Index n_created_points)
    -> tf::cut::arrangement_map_data<Index> {
  tf::cut::arrangement_map_data<Index> d;
  d.n_meshes = n_meshes;
  d.original_ids.resize(n_meshes);
  d.original_face_ids.resize(n_meshes);
  d.point_offsets.allocate(n_meshes + 1);
  d.poly_offsets.allocate(n_meshes + 1);
  d.point_offsets[0] = 0;
  d.poly_offsets[0] = 0;
  for (Index i = 0; i < n_meshes; ++i)
    apply_to_polygons(i, [&](const auto &p) {
      d.original_ids[i].reserve(p.points().size());
      d.original_face_ids[i].reserve(p.faces().size());
      d.point_offsets[i + 1] = d.point_offsets[i] + p.points().size();
      d.poly_offsets[i + 1] = d.poly_offsets[i] + p.faces().size();
    });

  d.original_map.allocate(d.point_offsets.back());
  const Index sentinel = static_cast<Index>(d.original_map.size());
  tf::parallel_fill(d.original_map, sentinel);

  tf::small_vector<tf::buffer<Index>, 10> face_masks(n_meshes);
  for (Index t = 0; t < n_meshes; ++t) {
    apply_to_polygons(t, [&](const auto &polygons) {
      face_masks[t].allocate(polygons.size());
      tf::parallel_fill(face_masks[t], true);
    });
  }

  d.original_offsets.allocate(n_meshes + 1);
  std::fill(d.original_offsets.begin(), d.original_offsets.end(), 0);

  tbb::task_group tg;

  for (auto zipped : tf::make_offset_block_range(
           fc.tag_offsets(), tf::zip(fc.descriptors(), fc.loops()))) {
    tg.run([&, zipped] {
      for (auto [desc, loop] : zipped) {
        auto &ids = d.original_ids[desc.tag];
        auto &curr = d.original_offsets[desc.tag + 1];
        face_masks[desc.tag][desc.object] = false;
        for (const auto &v : loop) {
          if (v.source == tf::intersect::graph::vertex_source::original) {
            auto flat = d.point_offsets[desc.tag] + v.id;
            if (d.original_map[flat] == sentinel) {
              d.original_map[flat] = curr++;
              ids.push_back(v.id);
            }
          }
        }
      }
    });
  }
  tg.wait();

  for (Index t = 0; t < n_meshes; ++t) {
    tg.run([&, t] {
      apply_to_polygons(t, [&, t](const auto &polygons) {
        auto &ids = d.original_ids[t];
        auto &curr = d.original_offsets[t + 1];
        auto off = d.point_offsets[t];
        for (auto [m, face_pair] :
             tf::zip(face_masks[t], tf::enumerate(polygons.faces()))) {
          if (!m)
            continue;
          auto [face_id, face] = face_pair;
          d.original_face_ids[t].push_back(face_id);
          for (auto v : face) {
            auto flat = off + Index(v);
            if (d.original_map[flat] == sentinel) {
              d.original_map[flat] = curr++;
              ids.push_back(Index(v));
            }
          }
        }
      });
    });
  }
  tg.wait();

  for (Index i = 0; i < n_meshes; ++i)
    d.original_offsets[i + 1] += d.original_offsets[i];

  d.original_face_offsets.allocate(n_meshes + 1);
  d.original_face_offsets[0] = 0;
  for (Index i = 0; i < n_meshes; ++i)
    d.original_face_offsets[i + 1] =
        d.original_face_offsets[i] +
        static_cast<Index>(d.original_face_ids[i].size());

  d.total_original_points = d.original_offsets.back();
  d.total_original_faces = d.original_face_offsets.back();

  tf::cut::discover_created(fc, n_created_points, d);

  return d;
}

/// Build partition map data for selected components.
///
/// Uses partition_ids to select which cut face loops and uncut polygons
/// to include per mesh. Builds both original and created vertex remapping.
/// Face IDs come from partition_ids — not stored here.
template <typename Index, typename Cuts, typename LabelType,
          typename ApplyToPolygons>
auto make_partition_map_data(
    const Cuts &fc, Index n_created_points,
    const tf::small_vector<tf::cut::partition_ids<Index>, 4> &pids,
    const tf::small_vector<LabelType, 4> &include_labels,
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

  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());

  // Pass 1: cut face loops — sequential (created vertices shared across meshes)
  for (Index t = 0; t < n_meshes; ++t) {
    auto &ids = d.original_ids[t];
    auto &curr = d.original_offsets[t + 1];
    auto off = d.point_offsets[t];
    auto label = include_labels[t];

    for (auto loop :
         tf::make_indirect_range(pids[t].cut_faces[label], loops_per_tag[t])) {
      for (const auto &v : loop) {
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
      }
    }
  }

  // Pass 2: uncut polygons — parallel (only original vertices, per-mesh)
  tbb::task_group tg;
  for (Index t = 0; t < n_meshes; ++t) {
    tg.run([&, t] {
      auto &ids = d.original_ids[t];
      auto &curr = d.original_offsets[t + 1];
      auto off = d.point_offsets[t];
      auto label = include_labels[t];

      apply_to_polygons(t, [&](const auto &polygons) {
        for (auto face : tf::make_indirect_range(pids[t].polygons[label],
                                                 polygons.faces())) {
          for (auto v : face) {
            auto flat = off + Index(v);
            if (d.original_map[flat] == sentinel_orig) {
              d.original_map[flat] = curr++;
              ids.push_back(Index(v));
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

/// Build splice map data for per-mesh independent construction.
///
/// Same vertex discovery as make_partition_map_data but original_offsets
/// stay zeroed (each mesh's IDs are 0-based) and created vertices are
/// tracked per mesh rather than shared.
template <typename Index, typename Cuts, typename LabelType,
          typename ApplyToPolygons>
auto make_splice_map_data(
    const Cuts &fc, Index n_created_points,
    const tf::small_vector<tf::cut::partition_ids<Index>, 4> &pids,
    const tf::small_vector<LabelType, 4> &include_labels,
    const ApplyToPolygons &apply_to_polygons)
    -> tf::cut::splice_map_data<Index> {
  auto n_meshes = static_cast<Index>(pids.size());
  tf::cut::splice_map_data<Index> d;
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

  d.created_maps.resize(n_meshes);
  d.created_ids.resize(n_meshes);
  d.created_counts.resize(n_meshes, 0);
  for (Index i = 0; i < n_meshes; ++i) {
    d.created_maps[i].allocate(n_created_points);
    tf::parallel_fill(d.created_maps[i], static_cast<Index>(n_created_points));
  }

  d.original_offsets.allocate(n_meshes + 1);
  std::fill(d.original_offsets.begin(), d.original_offsets.end(), 0);

  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());

  // Both passes fully parallel — each tag has independent created map
  tbb::task_group tg;
  for (Index t = 0; t < n_meshes; ++t) {
    tg.run([&, t] {
      auto &ids = d.original_ids[t];
      auto &curr = d.original_offsets[t + 1];
      auto off = d.point_offsets[t];
      auto label = include_labels[t];
      auto &cmap = d.created_maps[t];
      auto &cids = d.created_ids[t];
      auto &ccurr = d.created_counts[t];
      const Index sentinel_created = static_cast<Index>(n_created_points);

      // Cut face loops
      for (auto loop : tf::make_indirect_range(pids[t].cut_faces[label],
                                               loops_per_tag[t])) {
        for (const auto &v : loop) {
          if (v.source == tf::intersect::graph::vertex_source::original) {
            auto flat = off + v.id;
            if (d.original_map[flat] == sentinel_orig) {
              d.original_map[flat] = curr++;
              ids.push_back(v.id);
            }
          } else {
            if (cmap[v.id] == sentinel_created) {
              cmap[v.id] = ccurr++;
              cids.push_back(v.id);
            }
          }
        }
      }

      // Uncut polygons
      apply_to_polygons(t, [&](const auto &polygons) {
        for (auto face : tf::make_indirect_range(pids[t].polygons[label],
                                                 polygons.faces())) {
          for (auto v : face) {
            auto flat = off + Index(v);
            if (d.original_map[flat] == sentinel_orig) {
              d.original_map[flat] = curr++;
              ids.push_back(Index(v));
            }
          }
        }
      });
    });
  }
  tg.wait();

  // No prefix sum — offsets stay 0-based per mesh
  d.total_original_points = 0;
  for (Index i = 0; i < n_meshes; ++i)
    d.total_original_points += d.original_offsets[i + 1];

  return d;
}

/// Build embed map data for a single mesh (take all faces).
///
/// All original vertices referenced by tag's cut face loops and uncut
/// faces get compacted. All created vertices are used by direct offset.
template <typename Index, typename Cuts, typename Policy>
auto make_embed_map_data(const Cuts &fc,
                         const tf::polygons<Policy> &polygons, Index tag,
                         Index n_created_points)
    -> tf::cut::embed_map_data<Index> {
  tf::cut::embed_map_data<Index> d;
  auto n_pts = static_cast<Index>(polygons.points().size());
  d.original_map.allocate(n_pts);
  const Index sentinel = n_pts;
  tf::parallel_fill(d.original_map, sentinel);
  d.original_ids.reserve(n_pts);

  tf::buffer<bool> face_mask;
  face_mask.allocate(polygons.size());
  tf::parallel_fill(face_mask, true);

  Index curr = 0;
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());
  auto descs_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.descriptors());

  for (auto [desc, loop] : tf::zip(descs_per_tag[tag], loops_per_tag[tag])) {
    face_mask[desc.object] = false;
    for (const auto &v : loop) {
      if (v.source == tf::intersect::graph::vertex_source::original) {
        if (d.original_map[v.id] == sentinel) {
          d.original_map[v.id] = curr++;
          d.original_ids.push_back(v.id);
        }
      }
    }
  }

  for (Index i = 0; i < static_cast<Index>(polygons.size()); ++i) {
    if (!face_mask[i])
      continue;
    d.uncut_face_ids.push_back(i);
    for (auto v : polygons.faces()[i]) {
      auto vi = static_cast<Index>(v);
      if (d.original_map[vi] == sentinel) {
        d.original_map[vi] = curr++;
        d.original_ids.push_back(vi);
      }
    }
  }

  d.n_original_points = curr;
  tf::cut::discover_created(fc, n_created_points, d);

  return d;
}

} // namespace tf::cut
