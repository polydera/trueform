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
#include "../../core/checked.hpp"
#include "../../core/memory.hpp"
#include "../../core/none.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/projection_axes.hpp"
#include "./plane_triangulation_types.hpp"
#include "./refine_sizing.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// @brief March the splits out of the cut faces into the uncut mesh at the
///        sizing field's grade, so a refined side never meets an unrefined
///        one. Bounded rounds — each round's acceptances seed the next ring.
///
/// `scan` carries the (tag, u, v) edges whose splits have to reach their
/// uncut sharers; the caller seeds it and every round rewrites it from what
/// it accepted. `cut_mask` marks, per tag, the faces the arrangement already
/// holds.
///
/// The split currency is the caller's: `edge_splits(tag, a, b, params)`
/// appends the edge's positions in the order its own key gives them and
/// answers whether `a -> b` is that order, `emit_proposal(tag, a, b,
/// parameter, out)` states one proposal, `scan_key(proposal)` names the edge
/// it lands on, and `materialize(proposals)` returns how many of them became
/// splits.
template <typename Int, typename Index, typename Proposal,
          typename ApplyToForm, typename GetMeshPoint, typename EdgeSplits,
          typename EmitProposal, typename ScanKey, typename Materialize>
auto propagate_conforming_rings(
    tf::buffer<std::array<Index, 3>> &scan,
    const tf::core::std_vector<tf::buffer<char>> &cut_mask,
    const ApplyToForm &apply_to_form, const GetMeshPoint &get_mesh_point,
    const EdgeSplits &edge_splits, const EmitProposal &emit_proposal,
    const ScanKey &scan_key, const Materialize &materialize,
    tf::buffer<Proposal> &proposals) -> void {
  using param_t = typename tf::exact::meta<Int>::param_type;
  tf::buffer<std::array<Index, 2>> candidates;
  struct local {
    tf::buffer<std::array<double, 2>> points;
    tf::buffer<double> sizes;
    tf::buffer<double> edge_lengths;
    tf::buffer<std::size_t> edges;
    tf::buffer<std::uint32_t> parameters;
    tf::buffer<param_t> params;
    tf::buffer<char> forward;
    tf::buffer<Proposal> proposals;
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
        },
        tf::checked);
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
          state.forward.allocate(size);
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
            state.params.clear();
            const bool forward = edge_splits(tag, a, b, state.params);
            state.forward[edge] = char(forward);
            const std::size_t count = state.params.size();
            for (std::size_t offset = 0; offset < count; ++offset) {
              // The ring is a double-precision sizing curve, so a split
              // enters it at the ring's own resolution rather than the
              // split grid's. Half a bucket is the round-to-nearest bias;
              // both follow from the one bucket width.
              constexpr int bucket_shift =
                  tf::exact::meta<Int>::split_grid_bits -
                  tf::arrangement::ring_bucket_bits;
              const param_t position =
                  state.params[forward ? offset : count - 1 - offset];
              std::uint32_t parameter = std::uint32_t(
                  (position + (param_t(1) << (bucket_shift - 1))) >>
                  bucket_shift);
              if (!forward)
                parameter = tf::arrangement::ring_buckets - parameter;
              if (parameter == 0u ||
                  tf::arrangement::ring_buckets <= parameter)
                continue;
              if (state.edges.back() == edge &&
                  state.parameters.back() == parameter)
                continue;
              const double weight =
                  double(parameter) /
                  double(tf::arrangement::ring_buckets);
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
          tf::arrangement::cycle_sizing(
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
                    : tf::arrangement::ring_buckets;
            const Index a = Index(face[edge]);
            const Index b = Index(face[(edge + 1) % size]);
            const bool forward = state.forward[edge] != 0;
            tf::arrangement::dyadic_split(
                begin, end, state.points[point], state.points[next],
                state.sizes[point], state.sizes[next],
                [&](std::uint32_t parameter) {
                  const std::uint32_t canonical =
                      forward
                          ? parameter
                          : tf::arrangement::ring_buckets - parameter;
                  emit_proposal(
                      tag, a, b,
                      param_t(canonical)
                          << (tf::exact::meta<Int>::split_grid_bits -
                              tf::arrangement::ring_bucket_bits),
                      state.proposals);
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
      scan.push_back(scan_key(proposal));
    tbb::parallel_sort(scan.begin(), scan.end());
    scan.erase_till_end(std::unique(scan.begin(), scan.end()));
    if (materialize(proposals) == 0)
      break;
  }
}

} // namespace tf::arrangement
