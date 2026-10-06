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
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/boundary_rims.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_config.hpp"
#include "../hole_fill_result.hpp"
#include "../hole_fill_status.hpp"
#include "./fair_hole_patches.hpp"
#include "./fill_hole_rim.hpp"
#include "./make_hole_face_plans.hpp"
#include "./reject_overlapping_hole_rims.hpp"
#include "tbb/parallel_sort.h"
#include <array>
#include <cstddef>
#include <tuple>

namespace tf::fill {

/// Fill every rim and assemble the one result.
///
/// THE GRAIN IS THE GROUP: the rims run against each other through the
/// house partitioning primitive, so at most one table stands per worker,
/// and inside a rim its facet statement and the table's band stages fan out
/// again under their own checked cutoffs. Aggregation is in input order
/// because it is where a group's block-local mint names are rebased onto
/// the one mint lane the result publishes.
///
/// The face plans are built last, over the splits that SURVIVED — a refused
/// group publishes none — so the plan of a face two groups reached is one
/// plan, computed once, whatever became of either group. Fairing waits for
/// exactly that barrier: a patch is faired against the host as it finally
/// stands, which is why it is a pass of its own and not a step of the tier.
template <typename Int, typename Policy, typename MembershipPolicy,
          typename Index, typename RealT, typename Converter>
auto fill_hole_rims(const tf::polygons<Policy> &polygons,
                    const tf::face_membership_like<MembershipPolicy> &fm,
                    const tf::boundary_rims<Index> &rims,
                    const Converter &converter,
                    const tf::hole_fill_config &config,
                    tf::hole_fill_result<Index, RealT> &result) -> void {
  const Index n_points = Index(polygons.points().size());
  tf::buffer<Index> rejected;
  tf::fill::reject_overlapping_hole_rims(rims, rejected);

  struct local_t {
    tf::fill::hole_fill_scratch<Index, Int, RealT> scratch;
    tf::buffer<std::array<Index, 3>> group_triangles;
    tf::buffer<tf::point<RealT, 3>> group_minted;
    tf::buffer<tf::hole_split<Index>> group_splits;
    tf::buffer<std::array<Index, 3>> triangles;
    tf::buffer<tf::point<RealT, 3>> minted;
    tf::buffer<tf::hole_split<Index>> splits;
    tf::buffer<Index> triangle_sizes;
    tf::buffer<Index> mint_sizes;
    tf::buffer<Index> split_sizes;
    tf::buffer<tf::hole_fill_status> status;
    tf::buffer<tf::hole_refine_status> refined;
    tf::buffer<char> fairs;
    tf::buffer<Index> offending;
    tf::buffer<Index> group;
  };

  tf::buffer<char> asks;
  result.triangles.offsets_buffer().push_back(0);
  result.minted_points.offsets_buffer().push_back(0);

  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(rims.size()), std::tie(result, asks), local_t{},
      [&mesh = polygons, &fm, &boundary = rims, &converter, &config,
       &rejected, n_points](auto block, local_t &local) {
        local.triangles.clear();
        local.minted.clear();
        local.splits.clear();
        local.triangle_sizes.clear();
        local.mint_sizes.clear();
        local.split_sizes.clear();
        local.status.clear();
        local.refined.clear();
        local.fairs.clear();
        local.offending.clear();
        local.group.clear();
        for (auto slot : block) {
          const Index rim = Index(slot);
          local.group_triangles.clear();
          local.group_minted.clear();
          local.group_splits.clear();
          auto refined = tf::hole_refine_status::not_attempted;
          bool fairs = false;
          Index offending = Index(-1);
          tf::hole_fill_status status;
          if (rejected[std::size_t(rim)] != Index(-1)) {
            offending = rejected[std::size_t(rim)];
            status = tf::hole_fill_status::refused_invalid_rim;
          } else {
            status = tf::fill::fill_hole_rim(
                mesh, fm, boundary, rim, converter, config, n_points,
                local.scratch, local.group_triangles, local.group_minted,
                local.group_splits, refined, fairs, offending);
          }
          if (status != tf::hole_fill_status::filled) {
            local.group_triangles.clear();
            local.group_minted.clear();
            local.group_splits.clear();
            refined = tf::hole_refine_status::not_attempted;
            fairs = false;
          }
          local.triangle_sizes.push_back(Index(local.group_triangles.size()));
          local.mint_sizes.push_back(Index(local.group_minted.size()));
          local.split_sizes.push_back(Index(local.group_splits.size()));
          for (const auto &triangle : local.group_triangles)
            local.triangles.push_back(triangle);
          for (const auto &point : local.group_minted)
            local.minted.push_back(point);
          for (const auto &split : local.group_splits)
            local.splits.push_back(split);
          local.status.push_back(status);
          local.refined.push_back(refined);
          local.fairs.push_back(char(fairs));
          local.offending.push_back(offending);
          local.group.push_back(rim);
        }
      },
      [n_points](const local_t &local, auto &aggregated) {
        auto &[out, asked] = aggregated;
        std::size_t triangle = 0, mint = 0, split = 0;
        for (std::size_t g = 0; g < local.status.size(); ++g) {
          const Index base = Index(out.minted_points.data_buffer().size());
          const auto rebase = [base, n_points](Index corner) {
            return corner < n_points ? corner : Index(corner + base);
          };
          for (Index k = 0; k < local.mint_sizes[g]; ++k)
            out.minted_points.data_buffer().push_back(local.minted[mint++]);
          for (Index k = 0; k < local.triangle_sizes[g]; ++k) {
            const auto &corners = local.triangles[triangle++];
            out.triangles.data_buffer().push_back(
                {rebase(corners[0]), rebase(corners[1]), rebase(corners[2])});
          }
          for (Index k = 0; k < local.split_sizes[g]; ++k) {
            auto record = local.splits[split++];
            record.point = rebase(record.point);
            out.splits.push_back(record);
          }
          out.triangles.offsets_buffer().push_back(
              Index(out.triangles.data_buffer().size()));
          out.minted_points.offsets_buffer().push_back(
              Index(out.minted_points.data_buffer().size()));
          out.status.push_back(local.status[g]);
          out.refined.push_back(local.refined[g]);
          out.faired.push_back(tf::hole_fair_status::not_attempted);
          out.seam_max_angle.push_back(RealT(-1));
          asked.push_back(local.fairs[g]);
          const Index ticket = local.offending[g];
          out.offending.data_buffer().push_back(
              ticket == Index(-1) ? Index(-1) : local.group[g]);
          out.offending.data_buffer().push_back(ticket);
        }
      });

  tbb::parallel_sort(result.splits.begin(), result.splits.end(),
                     [](const tf::hole_split<Index> &a,
                        const tf::hole_split<Index> &b) {
                       if (a.face != b.face)
                         return a.face < b.face;
                       if (a.v0 != b.v0)
                         return a.v0 < b.v0;
                       if (a.v1 != b.v1)
                         return a.v1 < b.v1;
                       return a.parameter < b.parameter;
                     });
  tf::fill::make_hole_face_plans(polygons, tf::make_range(result.splits),
                                 result.plan_faces, result.plan_triangles);
  tf::fill::fair_hole_patches(polygons, fm, config.fairing, asks, result);
}

} // namespace tf::fill
