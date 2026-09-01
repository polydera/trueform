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
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "./plane_piece_key.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace tf::arrangement {

/// Materialize the complete definition span of every PA-owned final piece.
/// THE OWNERSHIP LAW makes the current span the whole truth: a fused group's
/// union concatenated every contributor's instances when its carriers
/// promoted, so nothing is restored from the immutable tier here.
/// Immutable-only pieces remain in the caller-owned prefix.
template <typename Index, typename Immutable, typename Current>
auto make_plane_final_piece_definitions(
    const Immutable &immutable, const Current &current, Index immutable_extent,
    tf::offset_block_buffer<
        Index, tf::intersect::graph::plane_edge_def<Index>> &output) -> bool {
  output.clear();
  if (immutable_extent != immutable.n_canon() ||
      immutable_extent < Index(0) || current.n_canon() < Index(0) ||
      current.n_canon() >
          std::numeric_limits<Index>::max() - immutable_extent ||
      std::size_t(current.n_canon()) ==
          std::numeric_limits<std::size_t>::max())
    return false;
  const auto n_groups = current.n_canon();
  if (n_groups == Index(0))
    return true;

  const auto immutable_definitions = immutable.edge_defs();
  const auto current_definitions = current.edge_defs();
  const auto max_size = std::size_t(std::numeric_limits<Index>::max());
  if (immutable_definitions.size() > max_size ||
      current_definitions.size() > max_size)
    return false;

  tf::buffer<char> valid;
  valid.allocate(std::size_t(n_groups));
  std::fill(valid.begin(), valid.end(), char(1));
  tf::buffer<Index> counts;
  counts.allocate(std::size_t(n_groups));
  std::fill(counts.begin(), counts.end(), Index(0));
  tf::parallel_for_each(
      tf::make_sequence_range(n_groups), [&](Index group) {
        const auto current_span = current.canon_group(group);
        if (current_span.size() == 0) {
          valid[std::size_t(group)] = char(0);
          return;
        }
        const auto key = plane_piece_key<Index>(current_span[0]);
        for (const auto &definition : current_span)
          if (definition.id != group ||
              plane_piece_key<Index>(definition) != key)
            valid[std::size_t(group)] = char(0);
        counts[std::size_t(group)] = Index(current_span.size());
      });
  if (!std::all_of(valid.begin(), valid.end(),
                   [](char value) { return value != 0; }))
    return false;
  auto &offsets = output.offsets_buffer();
  auto &definitions = output.data_buffer();
  offsets.allocate(std::size_t(n_groups) + 1);
  offsets[0] = Index(0);
  for (Index group = 0; group < n_groups; ++group) {
    const auto count = counts[std::size_t(group)];
    if (count > std::numeric_limits<Index>::max() -
                    offsets[std::size_t(group)]) {
      output.clear();
      return false;
    }
    offsets[std::size_t(group) + 1] =
        offsets[std::size_t(group)] + count;
  }
  definitions.allocate(std::size_t(offsets[std::size_t(n_groups)]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_groups), [&](Index group) {
        const auto current_span = current.canon_group(group);
        auto out = std::size_t(offsets[std::size_t(group)]);
        for (auto definition : current_span) {
          definition.id = immutable_extent + group;
          definitions[out++] = definition;
        }
        const auto begin =
            definitions.begin() + offsets[std::size_t(group)];
        std::sort(begin, definitions.begin() + offsets[std::size_t(group) + 1],
                  tf::intersect::graph::plane_def_instance_less<Index>);
      });
  return true;
}

} // namespace tf::arrangement
