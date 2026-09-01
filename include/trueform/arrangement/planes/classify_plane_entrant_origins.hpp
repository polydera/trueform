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
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "./plane_recovery_birth.hpp"
#include "tbb/parallel_sort.h"

#include <cstddef>
#include <utility>

namespace tf::arrangement {

/// Classify raw entrant groups in the persistent exact edge-origin space.
///
/// Immutable canonical groups occupy the fixed prefix. A key absent from both
/// that prefix and the retained PA index receives the next append-only suffix
/// id. The complete next catalog is staged beside the persistent input; a
/// caller publishes it only with the generation that attached those origins.
/// A rejected later phase therefore cannot turn an uncommitted entrant into a
/// known collapsed origin on its next attempt.
template <typename Index, typename Immutable, typename RawEntrants>
auto classify_plane_entrant_origins(
    const Immutable &immutable, const RawEntrants &raw_entrants,
    Index immutable_extent,
    const tf::buffer<tf::intersect::graph::plane_edge_def<Index>> &origin_defs,
    const tf::buffer<plane_birth_edge_record<Index>> &origin_index,
    tf::buffer<tf::intersect::graph::plane_edge_def<Index>> &next_origin_defs,
    tf::buffer<plane_birth_edge_record<Index>> &next_origin_index,
    tf::buffer<Index> &origin_of_group, tf::buffer<char> &fresh_origin)
    -> bool {
  if (immutable_extent != immutable.n_canon() ||
      origin_index.size() != origin_defs.size())
    return false;

  const auto n_groups = raw_entrants.n_canon();
  tf::buffer<Index> classified;
  tf::buffer<char> fresh;
  classified.allocate(std::size_t(n_groups));
  fresh.allocate(std::size_t(n_groups));
  tf::parallel_for_each(
      tf::make_sequence_range(n_groups),
      [&](Index group) {
        const auto span = raw_entrants.canon_group(group);
        const auto origin = find_plane_edge_origin(
            immutable, span[0], immutable_extent, origin_index);
        classified[std::size_t(group)] = origin;
        fresh[std::size_t(group)] = char(origin == Index(-1));
      },
      tf::checked);

  tf::buffer<Index> fresh_offsets;
  fresh_offsets.allocate(std::size_t(n_groups) + 1);
  fresh_offsets[0] = Index(0);
  for (Index group = 0; group < n_groups; ++group)
    fresh_offsets[std::size_t(group) + 1] =
        fresh_offsets[std::size_t(group)] +
        Index(fresh[std::size_t(group)] != char(0));
  const auto n_fresh = fresh_offsets[std::size_t(n_groups)];

  tf::buffer<tf::intersect::graph::plane_edge_def<Index>> staged_defs(
      origin_defs);
  tf::buffer<plane_birth_edge_record<Index>> staged_index(origin_index);
  if (n_fresh != Index(0)) {
    const auto old_size = origin_defs.size();
    const auto old_index_size = origin_index.size();
    staged_defs.reallocate(old_size + std::size_t(n_fresh));
    staged_index.reallocate(old_index_size + std::size_t(n_fresh));
    tf::parallel_for_each(
        tf::make_sequence_range(n_groups),
        [&](Index group) {
          if (!fresh[std::size_t(group)])
            return;
          const auto suffix = fresh_offsets[std::size_t(group)];
          const auto origin = immutable_extent + Index(old_size) + suffix;
          auto retained = raw_entrants.canon_group(group)[0];
          retained.id = origin;
          const auto key = plane_recovery_edge_key(retained);
          staged_defs[old_size + std::size_t(suffix)] = retained;
          staged_index[old_index_size + std::size_t(suffix)] = {
              key[0], key[1], key[2], key[3], origin};
          classified[std::size_t(group)] = origin;
        },
        tf::checked);
    tbb::parallel_sort(staged_index.begin(), staged_index.end());
  }

  next_origin_defs = std::move(staged_defs);
  next_origin_index = std::move(staged_index);
  origin_of_group = std::move(classified);
  fresh_origin = std::move(fresh);
  return true;
}

} // namespace tf::arrangement
