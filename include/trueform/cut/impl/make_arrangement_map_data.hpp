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
#include "../../core/views/enumerate.hpp"
#include "../../core/views/zip.hpp"
#include "../arrangement_map_data.hpp"
#include "../face_cuts.hpp"
#include "tbb/task_group.h"

namespace tf::cut {

/// Build arrangement map data: vertex ID maps, face masks, offset tables.
///
/// Walks cut face loops and uncut faces in parallel per mesh, assigns
/// contiguous IDs to original vertices on first encounter. Created
/// vertices (intersection points) are all used — offset after originals.
template <typename Index, typename ApplyToPolygons>
auto make_arrangement_map_data(
    const tf::face_cuts<Index> &fc,
    const ApplyToPolygons &apply_to_polygons, Index n_meshes)
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

  return d;
}

} // namespace tf::cut
