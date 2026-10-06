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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_result.hpp"
#include "../hole_fill_status.hpp"
#include "./hole_fairing_stencil.hpp"
#include "./hole_seam_angle.hpp"
#include "./solve_hole_fairing.hpp"
#include "./solve_hole_fairing_axis.hpp"
#include <array>
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// Fair every group that asked for it, and state every filled group's seam.
///
/// The plans are the authority this pass waits for: a group's stencil holds
/// the carrying faces as they finally stand, whichever groups' splits reached
/// them, which is why fairing runs after the canonical merge and not inside
/// the tier that produced the patch. Groups run against each other and each
/// is serial in itself; a group writes only the points it minted strictly
/// inside its own patch, so the one mint lane carries every group's answer
/// without a second pass over it.
///
/// The seam is stated for every group that holds a patch — off, refused or
/// faired alike — because it measures the patch that stands, not the solve.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename RealT>
auto fair_hole_patches(const tf::polygons<Policy> &polygons,
                       const tf::face_membership_like<MembershipPolicy> &fm,
                       bool fairing, const tf::buffer<char> &asks,
                       tf::hole_fill_result<Index, RealT> &result) -> void {
  const Index n_points = Index(polygons.points().size());

  struct local_t {
    tf::fill::hole_fairing_stencil<Index> stencil;
    tf::fill::hole_fairing_scratch<Index> solve;
    tf::buffer<tf::point<double, 3>> faired;
    tf::buffer<tf::point<RealT, 3>> staged;
    tf::buffer<std::array<Index, 3>> seam;
  };

  tf::parallel_for_each(
      tf::make_sequence_range(result.size()),
      [&polygons, &fm, &asks, &result, fairing, n_points](std::size_t group,
                                                          local_t &local) {
        if (result.status[group] != tf::hole_fill_status::filled)
          return;
        const bool complete = tf::fill::build_hole_fairing_stencil(
            polygons, fm, n_points, result.triangles[group], result.plan_faces,
            result.plan_triangles, result.minted_points.data_buffer(),
            local.solve.gather, local.stencil);

        auto state = tf::hole_fair_status::not_attempted;
        if (asks[group]) {
          if (!fairing)
            state = tf::hole_fair_status::fairing_off;
          else if (!complete ||
                   !tf::fill::solve_hole_fairing(local.stencil, local.solve,
                                                 local.faired))
            state = tf::hole_fair_status::fairing_refused;
          else {
            local.staged.clear();
            bool holds = true;
            for (const auto &point : local.faired) {
              const auto stated = point.template as<RealT>();
              for (std::size_t axis = 0; axis < 3; ++axis)
                holds = holds && std::isfinite(stated[axis]);
              local.staged.push_back(stated);
            }
            if (!holds)
              state = tf::hole_fair_status::fairing_refused;
            else {
              for (std::size_t k = 0; k < local.stencil.free_ids.size(); ++k) {
                const std::size_t slot =
                    std::size_t(local.stencil.free_ids[k]);
                const Index name = local.stencil.names[slot];
                result.minted_points
                    .data_buffer()[std::size_t(name - n_points)] =
                    local.staged[k];
                local.stencil.positions[slot] =
                    local.staged[k].template as<double>();
              }
              state = tf::hole_fair_status::faired;
            }
          }
        }
        result.faired[group] = state;
        result.seam_max_angle[group] =
            RealT(tf::fill::hole_seam_angle(local.stencil, local.seam));
      },
      local_t{});
}

} // namespace tf::fill
