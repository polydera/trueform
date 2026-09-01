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
#include "../../exact/meta.hpp"
#include "./materialize_plane_refinement_ring_splits.hpp"
#include "./plane_refinement_physical_split.hpp"
#include "./propagate_conforming_rings.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

struct plane_refinement_ring_result {
  /// Physical rows accepted after the initial whole-side table.
  std::size_t accepted_physical_rows = 0;
  /// Ring batches that reached exact proposal materialization.
  std::size_t materialization_rounds = 0;
  /// Outer sizing rings actually entered, including a final ring that found
  /// no candidate or proposal rows.
  std::size_t rings_run = 0;
};

/// Propagate refinement sizing through the four physical rings.
///
/// `cut_mask` is the one fixed mask shared with promotion and covers every face
/// already represented by the LA, including its entrant suffix. Accepted rows
/// preserve `physical` order. In a batch that accepts any row, the shared ring
/// kernel advances across every proposed edge: an existing adjacent split can
/// reject a new row without ending the sizing field at that edge.
template <typename Index, typename Int, typename CutMask, typename ApplyToForm,
          typename GetMeshPoint>
auto propagate_plane_refinement_rings(
    const CutMask &cut_mask, const ApplyToForm &apply_to_form,
    const GetMeshPoint &get_mesh_point,
    tf::buffer<plane_refinement_physical_split<Index, Int>> &physical)
    -> plane_refinement_ring_result {
  using param_t = typename tf::exact::meta<Int>::param_type;
  if (physical.size() == 0)
    return {};

  tf::buffer<std::array<Index, 3>> scan;
  scan.reserve(physical.size());
  std::array<Index, 3> previous{};
  bool has_previous = false;
  for (const auto &split : physical) {
    const auto edge = tf::arrangement::plane_refinement_physical_edge(split);
    if (!has_previous || edge != previous) {
      scan.push_back(edge);
      previous = edge;
      has_previous = true;
    }
  }

  plane_refinement_ring_result result;
  std::size_t last_accepted = 0;
  tf::buffer<plane_refinement_physical_split<Index, Int>> proposals;
  tf::arrangement::propagate_conforming_rings<Int>(
      scan, cut_mask, apply_to_form, get_mesh_point,
      [&physical](Index tag, Index a, Index b, tf::buffer<param_t> &params) {
        const bool forward = a < b;
        const std::array<Index, 3> edge{tag, forward ? a : b, forward ? b : a};
        auto row = std::lower_bound(
            physical.begin(), physical.end(), edge,
            [](const plane_refinement_physical_split<Index, Int> &split,
               const std::array<Index, 3> &key) {
              return tf::arrangement::plane_refinement_physical_edge(split) <
                     key;
            });
        for (; row != physical.end() &&
               tf::arrangement::plane_refinement_physical_edge(*row) == edge;
             ++row)
          params.push_back(row->parameter);
        return forward;
      },
      [](Index tag, Index a, Index b, param_t parameter,
         tf::buffer<plane_refinement_physical_split<Index, Int>> &out) {
        const bool forward = a < b;
        out.push_back({tag, forward ? a : b, forward ? b : a, parameter});
      },
      [](const plane_refinement_physical_split<Index, Int> &proposal) {
        return tf::arrangement::plane_refinement_physical_edge(proposal);
      },
      [&](tf::buffer<plane_refinement_physical_split<Index, Int>> &round) {
        ++result.materialization_rounds;
        const auto count =
            tf::arrangement::materialize_plane_refinement_ring_splits<Index,
                                                                      Int>(
                round, get_mesh_point, physical);
        last_accepted = count;
        result.accepted_physical_rows += count;
        return count;
      },
      proposals);
  if (result.materialization_rounds == 0) {
    result.rings_run = 1;
  } else {
    result.rings_run = result.materialization_rounds;
    if (last_accepted != 0 && result.materialization_rounds < 4)
      ++result.rings_run;
  }
  return result;
}

} // namespace tf::arrangement
