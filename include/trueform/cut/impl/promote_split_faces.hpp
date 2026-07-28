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

#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/constant.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../../topology/cdt_refiner.hpp"
#include "../../topology/topo_id.hpp"
#include "../detail/promote_split_faces_applied.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./collect_region_steiner_points.hpp"
#include "./inject_promoted_face_steiners.hpp"
#include "./make_promoted_face_candidates.hpp"
#include "./make_region_interior_points.hpp"
#include "./triangulate_region.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int, typename ApplyToForm,
          typename ApplyToFace, typename GetMeshPoint,
          typename IntersectionPoints, typename OriginalEdgeSplit>
auto promote_split_faces(
    const tf::face_regions<Index, Int> &regions, Index n_tags,
    const ApplyToForm &apply_to_form,
    const ApplyToFace &apply_to_face,
    const GetMeshPoint &get_mesh_point,
    const IntersectionPoints &intersection_points,
    Index n_intersection_points,
    const tf::buffer<OriginalEdgeSplit> &original_edge_splits,
    const tf::buffer<std::array<Index, 3>> &promoted_steiners,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits,
    const tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &merges,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>> &triangles,
    tf::buffer<Index> &failed,
    tf::buffer<tf::intersect::graph::face_descriptor<Index>> &promoted,
    tf::buffer<Index> &promoted_interior_counts,
    Index &promoted_interior_base,
    tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &pending_merges) -> void {
  struct local {
    tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
    tf::buffer<tf::intersect::graph::vertex<Index>> walk;
    tf::buffer<Index> no_holes;
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>>
        triangles;
    tf::buffer<tf::point<Int, 3>> interior_points;
    tf::buffer<Index> counts;
    tf::buffer<Index> interior_counts;
    tf::buffer<char> ok;
  };
  auto candidates = tf::cut::make_promoted_face_candidates(
      regions, n_tags, apply_to_form, original_edge_splits,
      promoted_steiners);
  auto make_emit = [&] {
    return [&, apply_to_face_copy = apply_to_face,
            get_mesh_point_copy = get_mesh_point](
               Index, const std::array<Index, 2> &candidate,
               local &state) {
      const tf::intersect::graph::face_descriptor<Index> descriptor{
          candidate[0], candidate[1]};
      state.walk.clear();
      apply_to_face_copy(candidate[0], candidate[1],
                         [&](const auto &face) {
                           for (std::size_t corner = 0;
                                corner < face.size(); ++corner)
                             state.walk.push_back({
                                 tf::intersect::graph::vertex_source::
                                     original,
                                 Index(face[corner]),
                                 {short(corner),
                                  tf::topo_type::vertex}});
                         });
      const std::size_t before = state.triangles.size();
      const bool ok = tf::cut::triangulate_region(
          regions, descriptor, tf::make_range(state.walk),
          tf::make_range(state.no_holes), apply_to_face_copy,
          get_mesh_point_copy, intersection_points, n_tags,
          n_intersection_points, extra_points, splits, merges,
          state.workspace, state.triangles, true);
      if (!ok)
        state.triangles.erase_till_end(
            state.triangles.begin() + std::ptrdiff_t(before));
      state.counts.push_back(
          Index(state.triangles.size() - before));
      state.interior_counts.push_back(Index(0));
      state.ok.push_back(char(ok));
    };
  };
  tf::cut::detail::promote_split_faces_applied(
      candidates, n_intersection_points, make_emit, local{},
      extra_points, ranges, triangles, failed, promoted,
      promoted_interior_counts, promoted_interior_base, pending_merges);
}

template <typename Index, typename Int, typename ApplyToForm,
          typename ApplyToFace, typename GetMeshPoint,
          typename IntersectionPoints, typename OriginalEdgeSplit>
auto promote_split_faces(
    const tf::face_regions<Index, Int> &regions, Index n_tags,
    const ApplyToForm &apply_to_form,
    const ApplyToFace &apply_to_face,
    const GetMeshPoint &get_mesh_point,
    const IntersectionPoints &intersection_points,
    Index n_intersection_points,
    const tf::buffer<OriginalEdgeSplit> &original_edge_splits,
    const tf::buffer<std::array<Index, 3>> &promoted_steiners,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits,
    const tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &merges,
    const tf::cdt_refine_config &config,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>> &triangles,
    tf::buffer<Index> &failed,
    tf::buffer<tf::intersect::graph::face_descriptor<Index>> &promoted,
    tf::buffer<Index> &promoted_interior_counts,
    Index &promoted_interior_base,
    tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &pending_merges) -> void {
  struct local {
    tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
    tf::cdt_refiner<Index, Int, Int> refiner;
    tf::buffer<tf::intersect::graph::vertex<Index>> walk;
    tf::buffer<Index> no_holes;
    tf::buffer<bool> splittable;
    tf::buffer<char> point_flags;
    tf::buffer<tf::point<Int, 3>> steiner_points;
    tf::buffer<tf::cut::detail::region_interior_point<Index, Int>>
        interior_vertices;
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>>
        triangles;
    tf::buffer<tf::point<Int, 3>> interior_points;
    tf::buffer<Index> counts;
    tf::buffer<Index> interior_counts;
    tf::buffer<char> ok;
  };
  auto candidates = tf::cut::make_promoted_face_candidates(
      regions, n_tags, apply_to_form, original_edge_splits,
      promoted_steiners);
  auto make_emit = [&] {
    return [&, apply_to_face_copy = apply_to_face,
            get_mesh_point_copy = get_mesh_point](
               Index candidate_index,
               const std::array<Index, 2> &candidate,
               local &state) {
      const tf::intersect::graph::face_descriptor<Index> descriptor{
          candidate[0], candidate[1]};
      state.walk.clear();
      apply_to_face_copy(candidate[0], candidate[1],
                         [&](const auto &face) {
                           for (std::size_t corner = 0;
                                corner < face.size(); ++corner)
                             state.walk.push_back({
                                 tf::intersect::graph::vertex_source::
                                     original,
                                 Index(face[corner]),
                                 {short(corner),
                                  tf::topo_type::vertex}});
                         });
      const std::size_t before = state.triangles.size();
      const std::size_t interior_before =
          state.interior_points.size();
      const auto steiner_range =
          candidates.steiner_ranges[std::size_t(candidate_index)];
      state.interior_vertices.clear();
      for (Index steiner = steiner_range[0];
           steiner < steiner_range[1]; ++steiner)
        state.interior_vertices.push_back(
            tf::cut::region_interior_point_of(
                promoted_steiners[std::size_t(steiner)][2],
                n_intersection_points, extra_points));
      if (tf::cut::triangulate_region(
              regions, descriptor, tf::make_range(state.walk),
              tf::make_range(state.no_holes), apply_to_face_copy,
              get_mesh_point_copy, intersection_points, n_tags,
              n_intersection_points, extra_points, splits, merges,
              state.workspace, state.triangles, true,
              tf::cut::detail::region_build::assemble)) {
        tf::cut::inject_promoted_face_steiners(
            steiner_range[0], steiner_range[1],
            promoted_steiners, n_intersection_points,
            extra_points, state.workspace);
        auto edges = tf::make_edges(tf::make_blocked_range<2>(
            tf::make_range(state.workspace.cons.data(),
                           state.workspace.cons.size())));
        state.splittable.allocate(edges.size());
        std::fill(state.splittable.begin(), state.splittable.end(), false);
        state.refiner.clear();
        if (state.refiner.build(
                tf::make_points(state.workspace.pts), edges,
                tf::make_constant_range(true, edges.size()),
                tf::make_range(state.splittable), config)) {
          state.steiner_points.clear();
          tf::cut::collect_region_steiner_points(
              state.workspace, state.refiner, state.point_flags,
              state.steiner_points);
          for (const auto &position : state.steiner_points) {
            // The id these points get depends on where the block lands in
            // the aggregate, so they name themselves by position in it.
            state.interior_vertices.push_back(
                {{tf::intersect::graph::vertex_source::created,
                  Index(-Index(state.interior_points.size()) - 1),
                  {0, tf::topo_type::face}},
                 position});
            state.interior_points.push_back(position);
          }
        }
      }
      const bool ok = tf::cut::triangulate_region(
          regions, descriptor, tf::make_range(state.walk),
          tf::make_range(state.no_holes), apply_to_face_copy,
          get_mesh_point_copy, intersection_points, n_tags,
          n_intersection_points, extra_points, splits, merges,
          state.workspace, state.triangles, true,
          tf::cut::detail::region_build::preserve,
          tf::make_range(state.interior_vertices));
      if (!ok) {
        state.interior_points.erase_till_end(
            state.interior_points.begin() +
            std::ptrdiff_t(interior_before));
        state.triangles.erase_till_end(
            state.triangles.begin() + std::ptrdiff_t(before));
      }
      state.counts.push_back(
          Index(state.triangles.size() - before));
      state.interior_counts.push_back(
          Index(state.interior_points.size() - interior_before));
      state.ok.push_back(char(ok));
    };
  };
  tf::cut::detail::promote_split_faces_applied(
      candidates, n_intersection_points, make_emit, local{},
      extra_points, ranges, triangles, failed, promoted,
      promoted_interior_counts, promoted_interior_base, pending_merges);
}

} // namespace tf::cut
