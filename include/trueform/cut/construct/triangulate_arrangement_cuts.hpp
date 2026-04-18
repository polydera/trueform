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
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../topology/delaunay_triangulator.hpp"
#include "./arrangement_map_data.hpp"

namespace tf::cut {

/// Triangulate cut face loops and produce triangle indices + labels.
///
/// For each cut face loop, projects to 2D using make_projector (int32),
/// triangulates with exact delaunay_triangulator, and maps vertex indices through
/// map_data.map_vertex.
/// Produces three parallel buffers: triangle indices, tag labels, face labels.
template <typename Int, typename Index, typename Range, typename MapVertex,
          typename MakeProjector>
auto triangulate_arrangement_cuts(const Range &zipped_descs_loops,
                                  const MakeProjector &make_projector,
                                  const MapVertex &map_vertex,
                                  tf::buffer<Index> &triangles,
                                  tf::buffer<Index> &tag_labels,
                                  tf::buffer<Index> &face_labels) {

  tf::generic_generate(zipped_descs_loops,
                       std::tie(triangles, tag_labels, face_labels),
                       std::make_pair(tf::small_vector<tf::point<Int, 2>, 10>{},
                                      tf::delaunay_triangulator<Index, Int>{}),
                       [&make_projector, &map_vertex](
                           const auto &pair, auto &buffers, auto &state) {
                         auto [desc, loop] = pair;
                         auto &[tri_buf, tag_buf, face_buf] = buffers;
                         auto &[pts, tri] = state;
                         auto projector = make_projector(desc);
                         pts.clear();
                         for (const auto &v : loop)
                           pts.push_back(projector(v));
                         tri.build(tf::make_points(pts));
                         auto n_tris = tri.indices_buffer().size() / 3;
                         for (auto id : tri.indices_buffer())
                           tri_buf.push_back(map_vertex(desc.tag, loop[id]));
                         for (std::size_t i = 0; i < n_tris; ++i) {
                           tag_buf.push_back(desc.tag);
                           face_buf.push_back(desc.object);
                         }
                       });
}

template <typename Int, typename Index, typename Range, typename MakeProjector>
auto triangulate_arrangement_cuts(
    const Range &zipped_descs_loops, const MakeProjector &make_projector,
    const tf::cut::arrangement_map_data<Index> &map_data,
    tf::buffer<Index> &triangles, tf::buffer<Index> &tag_labels,
    tf::buffer<Index> &face_labels) {
  triangulate_arrangement_cuts<Int>(
      zipped_descs_loops, make_projector,
      [&](auto tag, const auto &v) { return map_data.map_vertex(tag, v); },
      triangles, tag_labels, face_labels);
}

/// Triangulate selected cut face loops for one mesh partition.
///
/// Only produces triangle indices and face origins (no tag labels needed
/// since the caller knows which mesh this batch belongs to).
template <typename Int, typename Index, typename Range, typename MakeProjector,
          typename MapVertex>
auto triangulate_partition_cuts(const Range &zipped_descs_loops,
                                const MakeProjector &make_projector,
                                const MapVertex &map_vertex,
                                tf::buffer<Index> &triangles,
                                tf::buffer<Index> &face_origins) {
  tf::generic_generate(zipped_descs_loops, std::tie(triangles, face_origins),
                       std::make_pair(tf::small_vector<tf::point<Int, 2>, 10>{},
                                      tf::delaunay_triangulator<Index, Int>{}),
                       [&make_projector, &map_vertex](
                           const auto &pair, auto &buffers, auto &state) {
                         auto [desc, loop] = pair;
                         auto &[tri_buf, origin_buf] = buffers;
                         auto &[pts, tri] = state;
                         auto projector = make_projector(desc);
                         pts.clear();
                         for (const auto &v : loop)
                           pts.push_back(projector(v));
                         tri.build(tf::make_points(pts));
                         auto n_tris = tri.indices_buffer().size() / 3;
                         for (auto id : tri.indices_buffer())
                           tri_buf.push_back(map_vertex(desc.tag, loop[id]));
                         for (std::size_t i = 0; i < n_tris; ++i)
                           origin_buf.push_back(desc.object);
                       });
}

/// Triangulate partition cuts without tracking face origins.
template <typename Int, typename Index, typename Range, typename MakeProjector,
          typename MapVertex>
auto triangulate_partition_cuts(const Range &zipped_descs_loops,
                                const MakeProjector &make_projector,
                                const MapVertex &map_vertex,
                                tf::buffer<Index> &triangles) {
  tf::generic_generate(zipped_descs_loops, std::tie(triangles),
                       std::make_pair(tf::small_vector<tf::point<Int, 2>, 10>{},
                                      tf::delaunay_triangulator<Index, Int>{}),
                       [&make_projector, &map_vertex](
                           const auto &pair, auto &buffers, auto &state) {
                         auto [desc, loop] = pair;
                         auto &[tri_buf] = buffers;
                         auto &[pts, tri] = state;
                         auto projector = make_projector(desc);
                         pts.clear();
                         for (const auto &v : loop)
                           pts.push_back(projector(v));
                         tri.build(tf::make_points(pts));
                         for (auto id : tri.indices_buffer())
                           tri_buf.push_back(map_vertex(desc.tag, loop[id]));
                       });
}

} // namespace tf::cut
