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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/small_vector.hpp"
#include "./arrangement_map_data.hpp"
#include "tbb/task_group.h"
#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// Build the full-arrangement map data over the exposed triangle
/// stream.
///
/// The SLOT is the authority for what is uncut: a slot with
/// `plane == -1` is a face the arrangement never cut, so the source
/// form is its geometry; every other slot's triangles carry it. Stream
/// originals name FLAT ids, so a tag's point block stays pure — a
/// coincident cross-tag original is a created class at birth.
template <typename Index, typename Arrangement, typename ApplyToPolygons>
auto make_arrangement_stream_map_data(const Arrangement &arrangement,
                                      const ApplyToPolygons &apply_to_polygons)
    -> tf::arrangement::arrangement_map_data<Index> {
  const auto n_meshes = arrangement.n_tags();
  const auto &ga = arrangement.global();
  auto descriptors = ga.exposed_descriptors();
  auto face_slot_base = ga.face_slot_base();
  auto tris = ga.exposed_tris();

  tf::arrangement::arrangement_map_data<Index> d;
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

  d.original_offsets.allocate(n_meshes + 1);
  std::fill(d.original_offsets.begin(), d.original_offsets.end(), 0);

  // Every tag's triangles reference only created ids and that tag's own
  // flat originals, so the tags run in parallel.
  tbb::task_group tg;
  auto tag_offsets = ga.tag_offsets();
  for (Index t = 0; t < n_meshes; ++t)
    tg.run([&, t] {
      auto &ids = d.original_ids[t];
      auto &curr = d.original_offsets[t + 1];
      const auto base = d.point_offsets[t];
      for (Index e = tag_offsets[t]; e < tag_offsets[t + 1]; ++e)
        for (const auto &v : tris[std::size_t(e)])
          if (v.source == tf::intersect::graph::vertex_source::original) {
            if (d.original_map[v.id] == sentinel) {
              d.original_map[v.id] = curr++;
              ids.push_back(v.id - base);
            }
          }
    });
  tg.wait();

  for (Index t = 0; t < n_meshes; ++t)
    tg.run([&, t] {
      auto &ids = d.original_ids[t];
      auto &curr = d.original_offsets[t + 1];
      const auto base = d.point_offsets[t];
      const auto slot_base = Index(face_slot_base[std::size_t(t)]);
      apply_to_polygons(t, [&](const auto &polygons) {
        auto faces = polygons.faces();
        for (Index f = 0; f < static_cast<Index>(faces.size()); ++f) {
          if (descriptors[slot_base + f].plane != Index(-1))
            continue;
          d.original_face_ids[t].push_back(f);
          for (auto v : faces[f]) {
            const auto flat = base + Index(v);
            if (d.original_map[flat] == sentinel) {
              d.original_map[flat] = curr++;
              ids.push_back(Index(v));
            }
          }
        }
      });
    });
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

  const Index n_created =
      static_cast<Index>(arrangement.created_points().size());
  d.created_map.allocate(std::size_t(n_created));
  tf::parallel_fill(d.created_map, Index(0));
  tf::parallel_for_each(
      tris,
      [&](const auto &tri) {
        for (const auto &v : tri)
          if (v.source == tf::intersect::graph::vertex_source::created)
            d.created_map[std::size_t(v.id)] = Index(1);
      },
      tf::checked);
  d.created_ids.reserve(std::size_t(n_created));
  Index create_current = 0;
  for (Index i = 0; i < n_created; ++i)
    if (d.created_map[std::size_t(i)]) {
      d.created_map[std::size_t(i)] = create_current++;
      d.created_ids.push_back(i);
    } else {
      d.created_map[std::size_t(i)] = n_created;
    }
  d.total_created_points = create_current;

  return d;
}

} // namespace tf::arrangement
