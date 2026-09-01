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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_refinement_physical_split.hpp"
#include "./plane_refinement_plan.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace tf::arrangement {

/// Lift pristine-LA choices and promoted-face physical rows into one mixed
/// generation's exact split currency.
///
/// Initial choices use the immutable-group dense ticket. A physical row is
/// restated through every whole source side in the promoted entrant table and
/// uses the entrant-group ticket. Equal `(feature, parameter)` rows collapse
/// only after both sources name the same mixed canonical group.
template <typename Index, typename Int, typename InitialSplits,
          typename PhysicalRows, typename EntrantTables,
          typename EntrantDescriptors, typename MixedTables,
          typename ApplyToForm, typename VertexOffsets>
auto map_plane_refinement_splits(
    const InitialSplits &initial, const PhysicalRows &physical,
    const tf::buffer<Index> &immutable_group_ticket,
    const EntrantTables &entrants,
    const tf::buffer<Index> &entrant_group_ticket,
    const EntrantDescriptors &entrant_descriptors, Index entrant_face_base,
    Index entrant_plane_base, const MixedTables &mixed,
    const ApplyToForm &apply_to_form, Index n_flat,
    const VertexOffsets &vertex_offsets,
    tf::buffer<plane_refinement_split<Index, Int>> &output) -> bool {
  using param_t = typename tf::exact::meta<Int>::param_type;
  const param_t end_parameter = param_t(1) << tf::exact::meta<Int>::param_bits;

  output.clear();
  output.reserve(initial.size());
  for (const auto &split : initial) {
    if (split.source != plane_refinement_immutable_source ||
        split.group < Index(0) ||
        std::size_t(split.group) >= immutable_group_ticket.size())
      return false;
    const auto group = immutable_group_ticket[std::size_t(split.group)];
    if (group < Index(0) || group >= mixed.n_canon())
      return false;
    const auto ends = plane_recovery_flat_edge_ends(
        mixed.canon_group(group)[0], n_flat, vertex_offsets);
    const auto inverted = plane_recovery_flat_edge_inverted(ends);
    const std::array<Index, 2> feature{inverted ? ends[1] : ends[0],
                                       inverted ? ends[0] : ends[1]};
    output.push_back({feature, split.plane, group, split.parameter,
                      plane_refinement_current_source,
                      std::uint8_t(inverted)});
  }

  if (entrant_group_ticket.size() != std::size_t(entrants.n_canon()))
    return false;
  for (const auto group : entrant_group_ticket)
    if (group < Index(0) || group >= mixed.n_canon())
      return false;

  tf::generic_generate(
      tf::make_sequence_range(entrants.n_canon()), output,
      [&](Index raw_group,
          tf::buffer<plane_refinement_split<Index, Int>> &rows) {
        const auto mixed_group = entrant_group_ticket[std::size_t(raw_group)];
        const auto ends = plane_recovery_flat_edge_ends(
            mixed.canon_group(mixed_group)[0], n_flat, vertex_offsets);
        const auto inverted = plane_recovery_flat_edge_inverted(ends);
        const std::array<Index, 2> feature{inverted ? ends[1] : ends[0],
                                           inverted ? ends[0] : ends[1]};
        for (const auto &def : entrants.canon_group(raw_group)) {
          if (!(def.flags & tf::intersect::graph::plane_edge_whole_side_flag))
            continue;
          const auto entrant = def.face - entrant_face_base;
          const auto &descriptor = entrant_descriptors[std::size_t(entrant)];
          apply_to_form(Index(descriptor.tag), [&](const auto &form) {
            const auto face = form.faces()[descriptor.object];
            const auto side = std::size_t(def.side);
            const auto a = Index(face[side]);
            const auto b = Index(face[(side + 1) % face.size()]);
            const bool physical_reversed = b < a;
            const Index u = physical_reversed ? b : a;
            const Index v = physical_reversed ? a : b;
            const std::array<Index, 3> edge{Index(descriptor.tag), u, v};
            auto row = std::lower_bound(
                physical.begin(), physical.end(), edge,
                [](const auto &split, const std::array<Index, 3> &key) {
                  return plane_refinement_physical_edge(split) < key;
                });
            const bool reverse =
                bool(def.flags &
                     tf::intersect::graph::plane_edge_reversed_flag) !=
                physical_reversed;
            for (; row != physical.end() &&
                   plane_refinement_physical_edge(*row) == edge;
                 ++row) {
              const auto parameter =
                  reverse ? end_parameter - row->parameter : row->parameter;
              rows.push_back({feature, entrant_plane_base + entrant,
                              mixed_group, parameter,
                              plane_refinement_current_source,
                              std::uint8_t(inverted)});
            }
          });
        }
      },
      tf::checked);

  tbb::parallel_sort(
      output.begin(), output.end(),
      [](const plane_refinement_split<Index, Int> &a,
         const plane_refinement_split<Index, Int> &b) {
        return std::tie(a.feature, a.parameter, a.group, a.plane) <
               std::tie(b.feature, b.parameter, b.group, b.plane);
      });
  for (std::size_t row = 1; row < output.size(); ++row)
    if (output[row - 1].feature == output[row].feature &&
        output[row - 1].parameter == output[row].parameter &&
        output[row - 1].group != output[row].group)
      return false;
  output.erase_till_end(
      std::unique(output.begin(), output.end(),
                  [](const plane_refinement_split<Index, Int> &a,
                     const plane_refinement_split<Index, Int> &b) {
                    return a.feature == b.feature && a.parameter == b.parameter;
                  }));
  return true;
}

} // namespace tf::arrangement
