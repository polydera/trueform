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

#include "../../core/algorithm/sequenced_generate.hpp"
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

  tf::sequenced_generate(zipped_descs_loops,
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

/// Fold-aware overload: coincident stacks triangulate once.
///
/// `fold_of[loop_id] = {survivor, reversed}` (or `{-1, 0}` for live
/// loops, see @ref tf::cut::make_loop_fold_map). A dead loop
/// re-triangulates the SURVIVOR's loop with the survivor's projector —
/// the CDT is deterministic, so this reproduces the survivor's
/// triangulation exactly — maps vertices through its OWN tag (identity
/// equality is what made it a fold), emits its own labels, and flips
/// each triangle's winding when `reversed`. `descs` and `loops` must be
/// the FULL ranges: item index == loop id.
template <typename Int, typename Index, typename Descs, typename Loops,
          typename MakeProjector, typename MapVertex>
auto triangulate_arrangement_cuts(
    Descs descs, Loops loops, const MakeProjector &make_projector,
    const MapVertex &map_vertex,
    const tf::buffer<std::array<Index, 2>> &fold_of,
    tf::buffer<Index> &triangles, tf::buffer<Index> &tag_labels,
    tf::buffer<Index> &face_labels) {
  tf::sequenced_generate(
      tf::enumerate(tf::zip(descs, loops)),
      std::tie(triangles, tag_labels, face_labels),
      std::make_pair(tf::small_vector<tf::point<Int, 2>, 10>{},
                     tf::delaunay_triangulator<Index, Int>{}),
      [&make_projector, &map_vertex, &fold_of, descs,
       loops](const auto &item, auto &buffers, auto &state) {
        auto &&[loop_id, pair] = item;
        auto [desc, own_loop] = pair;
        auto &[tri_buf, tag_buf, face_buf] = buffers;
        auto &[pts, tri] = state;
        const auto fold = fold_of[static_cast<std::size_t>(loop_id)];
        const bool dead = fold[0] != Index(-1);
        const auto &src_desc = dead ? descs[fold[0]] : desc;
        auto loop = dead ? loops[fold[0]] : own_loop;
        auto projector = make_projector(src_desc);
        pts.clear();
        for (const auto &v : loop)
          pts.push_back(projector(v));
        tri.build(tf::make_points(pts));
        const auto &ids = tri.indices_buffer();
        const auto n_tris = ids.size() / 3;
        if (dead && fold[1]) {
          for (std::size_t t = 0; t < n_tris; ++t) {
            tri_buf.push_back(map_vertex(desc.tag, loop[ids[3 * t]]));
            tri_buf.push_back(map_vertex(desc.tag, loop[ids[3 * t + 2]]));
            tri_buf.push_back(map_vertex(desc.tag, loop[ids[3 * t + 1]]));
          }
        } else {
          for (auto id : ids)
            tri_buf.push_back(map_vertex(desc.tag, loop[id]));
        }
        for (std::size_t i = 0; i < n_tris; ++i) {
          tag_buf.push_back(desc.tag);
          face_buf.push_back(desc.object);
        }
      });
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
  tf::sequenced_generate(zipped_descs_loops, std::tie(triangles, face_origins),
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

/// Fold-aware partition overload: coincident stacks triangulate once
/// (see the fold-aware @ref triangulate_arrangement_cuts).
template <typename Int, typename Index, typename Descs, typename Loops,
          typename MakeProjector, typename MapVertex>
auto triangulate_partition_cuts(
    Descs descs, Loops loops, const MakeProjector &make_projector,
    const MapVertex &map_vertex,
    const tf::buffer<std::array<Index, 2>> &fold_of,
    tf::buffer<Index> &triangles, tf::buffer<Index> &face_origins) {
  tf::sequenced_generate(
      tf::enumerate(tf::zip(descs, loops)),
      std::tie(triangles, face_origins),
      std::make_pair(tf::small_vector<tf::point<Int, 2>, 10>{},
                     tf::delaunay_triangulator<Index, Int>{}),
      [&make_projector, &map_vertex, &fold_of, descs,
       loops](const auto &item, auto &buffers, auto &state) {
        auto &&[loop_id, pair] = item;
        auto [desc, own_loop] = pair;
        auto &[tri_buf, origin_buf] = buffers;
        auto &[pts, tri] = state;
        const auto fold = fold_of[static_cast<std::size_t>(loop_id)];
        const bool dead = fold[0] != Index(-1);
        const auto &src_desc = dead ? descs[fold[0]] : desc;
        auto loop = dead ? loops[fold[0]] : own_loop;
        auto projector = make_projector(src_desc);
        pts.clear();
        for (const auto &v : loop)
          pts.push_back(projector(v));
        tri.build(tf::make_points(pts));
        const auto &ids = tri.indices_buffer();
        const auto n_tris = ids.size() / 3;
        if (dead && fold[1]) {
          for (std::size_t t = 0; t < n_tris; ++t) {
            tri_buf.push_back(map_vertex(desc.tag, loop[ids[3 * t]]));
            tri_buf.push_back(map_vertex(desc.tag, loop[ids[3 * t + 2]]));
            tri_buf.push_back(map_vertex(desc.tag, loop[ids[3 * t + 1]]));
          }
        } else {
          for (auto id : ids)
            tri_buf.push_back(map_vertex(desc.tag, loop[id]));
        }
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
  tf::sequenced_generate(zipped_descs_loops, std::tie(triangles),
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
