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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "../detail/refine_sizing.hpp"
#include "./make_original_edge_splits.hpp"
#include "./materialize_region_split_proposals.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::cut {

template <typename Index, typename Int, typename ApplyToForm,
          typename GetMeshPoint, typename IntersectionPoints,
          typename OriginalEdgeSplit>
auto propagate_conforming_rings(
    const tf::face_regions<Index, Int> &regions, Index n_tags,
    Index n_intersection_points, const ApplyToForm &apply_to_form,
    const GetMeshPoint &get_mesh_point,
    const IntersectionPoints &intersection_points,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> &splits,
    tf::buffer<OriginalEdgeSplit> &original_edge_splits) -> void {
  auto descriptors = regions.descriptors();
  tf::core::std_vector<tf::buffer<char>> cut_mask(
      static_cast<std::size_t>(n_tags));
  for (Index tag = 0; tag < n_tags; ++tag) {
    apply_to_form(tag, [&](const auto &form) {
      cut_mask[std::size_t(tag)].allocate(form.faces().size());
    });
    std::fill(cut_mask[std::size_t(tag)].begin(),
              cut_mask[std::size_t(tag)].end(), char(0));
  }
  for (Index region = 0; region < Index(descriptors.size()); ++region)
    if (descriptors[region].tag != Index(-1))
      cut_mask[std::size_t(descriptors[region].tag)]
              [std::size_t(descriptors[region].object)] = 1;
  for (Index tag = 0; tag < n_tags; ++tag)
    for (auto face : regions.deleted(tag))
      cut_mask[std::size_t(tag)][std::size_t(face)] = 1;

  tf::buffer<std::array<Index, 3>> scan;
  for (const auto &split : original_edge_splits)
    scan.push_back({split.tag, split.u, split.v});
  tbb::parallel_sort(scan.begin(), scan.end());
  scan.erase_till_end(std::unique(scan.begin(), scan.end()));

  tf::buffer<std::array<Index, 2>> candidates;
  tf::buffer<tf::cut::detail::region_split_proposal<Index, Int>>
      proposals;
  struct local {
    tf::buffer<std::array<double, 2>> points;
    tf::buffer<double> sizes;
    tf::buffer<double> edge_lengths;
    tf::buffer<std::size_t> edges;
    tf::buffer<std::uint32_t> parameters;
    tf::buffer<tf::cut::detail::region_split_proposal<Index, Int>> proposals;
  };
  for (int ring = 0; ring < 4; ++ring) {
    candidates.clear();
    auto emit_candidate =
        [&](const std::array<Index, 3> &edge,
            tf::buffer<std::array<Index, 2>> &output) {
      apply_to_form(edge[0], [&](const auto &form) {
        for (auto face_id : form.face_membership()[edge[1]]) {
          if (cut_mask[std::size_t(edge[0])][std::size_t(face_id)])
            continue;
          auto face = form.faces()[face_id];
          const Index size = Index(face.size());
          for (Index face_edge = 0; face_edge < size; ++face_edge) {
            const Index a = Index(face[std::size_t(face_edge)]);
            const Index b =
                Index(face[std::size_t((face_edge + 1) % size)]);
            if ((a == edge[1] && b == edge[2]) ||
                (a == edge[2] && b == edge[1])) {
              output.push_back({edge[0], Index(face_id)});
              break;
            }
          }
        }
      });
    };
    if (scan.size() < 512) {
      for (const auto &edge : scan)
        emit_candidate(edge, candidates);
    } else {
      struct discovery_local {
        tf::buffer<std::array<Index, 2>> candidates;
      };
      tf::blocked_reduce_sequenced_aggregate(
          tf::make_range(scan), tf::none, discovery_local{},
          [&](auto &&range, discovery_local &state) {
            for (const auto &edge : range)
              emit_candidate(edge, state.candidates);
          },
          [&](const discovery_local &state, const tf::none_t &) {
            tf::core::append(state.candidates, candidates);
          });
    }
    tbb::parallel_sort(candidates.begin(), candidates.end());
    candidates.erase_till_end(
        std::unique(candidates.begin(), candidates.end()));
    if (candidates.size() == 0)
      break;

    proposals.clear();
    auto task = [&](auto &&range, local &state) {
      auto get_mesh_point_copy = get_mesh_point;
      for (const auto &candidate : range) {
        const Index tag = candidate[0];
        const Index face_id = candidate[1];
        apply_to_form(tag, [&](const auto &form) {
          auto face = form.faces()[face_id];
          const std::size_t size = face.size();
          const auto point_0 =
              get_mesh_point_copy(tag, Index(face[0]));
          const auto point_1 =
              get_mesh_point_copy(tag, Index(face[1]));
          const auto point_2 =
              get_mesh_point_copy(tag, Index(face[2]));
          const auto axes =
              tf::exact::projection_axes(point_0, point_1, point_2);
          state.points.clear();
          state.edges.clear();
          state.parameters.clear();
          for (std::size_t edge = 0; edge < size; ++edge) {
            const Index a = Index(face[edge]);
            const Index b = Index(face[(edge + 1) % size]);
            const auto a_point = get_mesh_point_copy(tag, a);
            const auto b_point = get_mesh_point_copy(tag, b);
            const std::array<double, 2> projected_a{
                double(a_point[axes.first]),
                double(a_point[axes.second])};
            const std::array<double, 2> projected_b{
                double(b_point[axes.first]),
                double(b_point[axes.second])};
            state.points.push_back(projected_a);
            state.edges.push_back(edge);
            state.parameters.push_back(0);
            const tf::intersect::graph::vertex<Index> a_vertex{
                tf::intersect::graph::vertex_source::original,
                a,
                {0, tf::topo_type::vertex}};
            const tf::intersect::graph::vertex<Index> b_vertex{
                tf::intersect::graph::vertex_source::original,
                b,
                {0, tf::topo_type::vertex}};
            const auto a_key = tf::cut::detail::region_vertex_key(
                n_tags, tag, a_vertex);
            const auto b_key = tf::cut::detail::region_vertex_key(
                n_tags, tag, b_vertex);
            const bool forward = a_key < b_key;
            const auto split_range =
                tf::cut::detail::find_region_edge_splits(
                    tf::cut::detail::region_edge_key(a_key, b_key),
                    splits);
            for (Index offset = 0;
                 offset < split_range.second - split_range.first;
                 ++offset) {
              const Index split =
                  forward ? split_range.first + offset
                          : split_range.second - 1 - offset;
              // The ring is a double-precision sizing curve, so a split
              // enters it at the ring's own resolution rather than the
              // split grid's. Half a bucket is the round-to-nearest bias;
              // both follow from the one bucket width.
              using param_t = typename tf::exact::meta<Int>::param_type;
              constexpr int bucket_shift =
                  tf::exact::meta<Int>::split_grid_bits -
                  tf::cut::detail::ring_bucket_bits;
              std::uint32_t parameter = std::uint32_t(
                  (splits[std::size_t(split)].param +
                   (param_t(1) << (bucket_shift - 1))) >>
                  bucket_shift);
              if (!forward)
                parameter = tf::cut::detail::ring_buckets - parameter;
              if (parameter == 0u ||
                  tf::cut::detail::ring_buckets <= parameter)
                continue;
              if (state.edges.back() == edge &&
                  state.parameters.back() == parameter)
                continue;
              const double weight =
                  double(parameter) / double(tf::cut::detail::ring_buckets);
              state.points.push_back(
                  {projected_a[0] +
                       weight * (projected_b[0] - projected_a[0]),
                   projected_a[1] +
                       weight * (projected_b[1] - projected_a[1])});
              state.edges.push_back(edge);
              state.parameters.push_back(parameter);
            }
          }
          const std::size_t point_count = state.points.size();
          tf::cut::detail::cycle_sizing(
              state.points, false, state.sizes, state.edge_lengths);
          auto no_local_feature_size =
              [](const std::array<double, 2> &) { return 1e300; };
          auto link_rows = form.manifold_edge_link()[face_id];
          for (std::size_t point = 0; point < point_count; ++point) {
            const std::size_t next = (point + 1) % point_count;
            const std::size_t edge = state.edges[point];
            if (!link_rows[edge].is_simple())
              continue;
            const std::uint32_t begin = state.parameters[point];
            const std::uint32_t end =
                next != 0 && state.edges[next] == edge
                    ? state.parameters[next]
                    : tf::cut::detail::ring_buckets;
            const Index a = Index(face[edge]);
            const Index b = Index(face[(edge + 1) % size]);
            const tf::intersect::graph::vertex<Index> a_vertex{
                tf::intersect::graph::vertex_source::original,
                a,
                {0, tf::topo_type::vertex}};
            const tf::intersect::graph::vertex<Index> b_vertex{
                tf::intersect::graph::vertex_source::original,
                b,
                {0, tf::topo_type::vertex}};
            const bool forward =
                tf::cut::detail::region_vertex_key(
                    n_tags, tag, a_vertex) <
                tf::cut::detail::region_vertex_key(
                    n_tags, tag, b_vertex);
            tf::cut::detail::dyadic_split(
                begin, end, state.points[point], state.points[next],
                state.sizes[point], state.sizes[next],
                [&](std::uint32_t parameter) {
                  const std::uint32_t canonical =
                      forward ? parameter
                              : tf::cut::detail::ring_buckets - parameter;
                  state.proposals.push_back(
                      {{a_vertex, b_vertex},
                       tag,
                       typename tf::exact::meta<Int>::param_type(canonical)
                           << (tf::exact::meta<Int>::split_grid_bits -
                               tf::cut::detail::ring_bucket_bits)});
                },
                no_local_feature_size);
          }
        });
      }
    };
    auto aggregate = [&](const local &state, const tf::none_t &) {
      tf::core::append(state.proposals, proposals);
    };
    tf::blocked_reduce_sequenced_aggregate(
        tf::make_range(candidates), tf::none, local{}, task, aggregate);
    if (proposals.size() == 0)
      break;

    scan.clear();
    for (const auto &proposal : proposals)
      scan.push_back(
          {proposal.tag, proposal.edge[0].id, proposal.edge[1].id});
    tbb::parallel_sort(scan.begin(), scan.end());
    scan.erase_till_end(std::unique(scan.begin(), scan.end()));
    if (tf::cut::materialize_region_split_proposals(
            n_tags, n_intersection_points, proposals,
            intersection_points, get_mesh_point, extra_points,
            splits) == 0)
      break;
  }
  tf::cut::make_original_edge_splits(
      splits, n_tags, original_edge_splits);
}

} // namespace tf::cut
