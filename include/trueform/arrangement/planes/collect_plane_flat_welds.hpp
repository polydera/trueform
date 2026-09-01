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
#include "./plane_flat_weld.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// Elect one provisional input anchor for every CDT output reached by an
/// input, and state every other input in that output as weld evidence.
///
/// The CDT's input-to-output map is the authority: projection twins may be
/// distinct in exact 3D and must not be rediscovered by comparing positions.
/// Created identities outrank originals and ties use the lowest flat key. An
/// output introduced by resolve mode legitimately has no input anchor and is
/// left as `-1`; an emission caller decides whether that is admissible.
///
/// `ends` are the anchors — the identities this plane names — and they are the
/// leading prepared inputs. `n_prepared` is everything the triangulation was
/// given, so a caller that also fed anonymous sites states them here: those
/// inputs claim no anchor and produce no weld evidence.
template <typename Index, typename Ends, typename Forward,
          typename OutputToInput>
auto collect_plane_flat_welds(const Ends &ends, const Forward &forward,
                              const OutputToInput &output_to_input,
                              Index n_prepared, Index n_final, Index n_flat,
                              Index plane, tf::buffer<Index> &representatives,
                              tf::buffer<plane_flat_weld<Index>> &welds)
    -> bool {
  const auto n_input = Index(ends.size());
  Index n_created = 0;
  for (Index output = 0; output < n_final; ++output)
    n_created += output_to_input[std::size_t(output)] == n_prepared ? Index(1)
                                                                    : Index(0);
  if (n_prepared == n_input && n_final == n_input && n_created == Index(0)) {
    representatives.allocate(std::size_t(n_final));
    for (Index output = 0; output < n_final; ++output)
      representatives[std::size_t(output)] =
          output_to_input[std::size_t(output)];
    return true;
  }
  representatives.allocate(std::size_t(n_final));
  std::fill(representatives.begin(), representatives.end(), Index(-1));
  auto prefers = [&](Index candidate, Index standing) {
    const auto candidate_created = ends[std::size_t(candidate)] >= n_flat;
    const auto standing_created = ends[std::size_t(standing)] >= n_flat;
    return candidate_created != standing_created
               ? candidate_created
               : ends[std::size_t(candidate)] < ends[std::size_t(standing)];
  };
  for (Index input = 0; input < n_input; ++input) {
    const auto output = forward[std::size_t(input)];
    if (output < 0 || n_final <= output)
      return false;
    auto &representative = representatives[std::size_t(output)];
    if (representative == Index(-1) || prefers(input, representative))
      representative = input;
  }
  for (Index input = 0; input < n_input; ++input) {
    const auto representative =
        representatives[std::size_t(forward[std::size_t(input)])];
    if (representative == input)
      continue;
    welds.push_back(
        {ends[std::size_t(input)], ends[std::size_t(representative)], plane});
  }
  return true;
}

} // namespace tf::arrangement
