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
#include "../../core/none.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./advance_plane_wave.hpp"
#include "./plane_arrangement_census.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_round_evidence.hpp"
#include "./update_plane_failures.hpp"

#include <array>
#include <cstddef>

namespace tf::arrangement {

/// What one closed round says about the loop: it is over, its frontier must be
/// triangulated again, this arrangement cannot publish what it holds, or the
/// round was handed back unconsumed because what it is about to state reaches
/// a face no tier of this arrangement has named.
enum class plane_round_result { done, retry, aborted, entrance };

/// WAVE. Close one round: commit its census, retire the failures it produced,
/// and let the planes that still refuse state what they saw. A round that
/// refuses nothing and welds nothing is the end of the loop, and so is a wave
/// that changes no carrier's constraint set — the planes still refusing then
/// keep the failure this round published for them. A SPLIT is not the only
/// change that counts: a substitution reaching a refusal's rows changes the
/// set that refusal never triangulated, and the wave publishes it. A weld
/// needs no refusal either: a triangulation that succeeded may still report
/// two identities it could not tell apart, and that statement is the wave's.
template <typename Index, typename Int, typename World, typename Planes,
          typename VertexOffsets, typename ExactPoint, typename NamePoint,
          typename StateEntrance = tf::none_t>
auto close_plane_round(
    const World &world, const Planes &planes,
    plane_round_evidence<Index, Int> &evidence,
    tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    tf::buffer<Index> &plane_ticket, tf::buffer<Index> &group_router,
    tf::buffer<std::array<Index, 3>> &merges, tf::buffer<Index> &created_class,
    tf::buffer<typename tf::exact::meta<Int>::param_type> &class_t,
    tf::buffer<plane_recovery_name<
        Index, typename tf::exact::meta<Int>::param_type>> &class_name,
    const VertexOffsets &vertex_offsets, Index n_flat, Index base_created,
    const ExactPoint &exact_point, const NamePoint &name_point,
    tf::buffer<Index> &failed, plane_arrangement_census &census,
    tf::buffer<Index> &frontier,
    const StateEntrance &state_entrance = tf::none) -> plane_round_result {
  evidence.census.refusals += evidence.refused.size();
  census += evidence.census;
  census.planes = std::size_t(world.n_planes());
  if (failed.size() != 0 || evidence.refused.size() != 0) {
    tf::buffer<Index> produced;
    produced.reserve(planes.size());
    std::size_t refused_at = 0;
    for (const auto plane : planes) {
      while (refused_at < evidence.refused.size() &&
             evidence.refused[refused_at] < plane)
        ++refused_at;
      if (refused_at < evidence.refused.size() &&
          evidence.refused[refused_at] == plane) {
        ++refused_at;
        continue;
      }
      produced.push_back(plane);
    }
    if (!update_plane_failures(failed, produced, evidence.refused))
      return plane_round_result::aborted;
    census.failed_planes = failed.size();
  }
  if (evidence.refused.size() == 0 && evidence.welds.size() == 0)
    return plane_round_result::done;
  const auto wave = advance_plane_wave(
      world, evidence, local_tables, plane_ticket, group_router, merges,
      created_class, class_t, class_name, vertex_offsets, n_flat, base_created,
      exact_point, name_point, census, frontier, state_entrance);
  census.created = created_class.size();
  if (wave == plane_wave_result::unsupported)
    return plane_round_result::aborted;
  // a caller that states no entrance can never be answered with one
  if (wave == plane_wave_result::entrance)
    return plane_round_result::entrance;
  return wave == plane_wave_result::stated ? plane_round_result::retry
                                           : plane_round_result::done;
}

} // namespace tf::arrangement
