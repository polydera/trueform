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

#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_statement.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// Elect one proposal that owns each wave class's birth position.
///
/// Statements and proposals are stated in one breath
/// (@ref tf::arrangement::state_plane_refusal), so the two carriers close into
/// the same name classes. Each independent producer marks its local birth
/// candidate; after equal names meet, the first candidate in the canonical
/// proposal order is the class authority. The engine records that position
/// without asking which producer stated it.
///
/// False rejects the call when the two class carriers disagree, an edge is
/// not stated in one of the two table tiers, or a class has no birth
/// candidate. All indict the producer, not the geometry it describes.
template <typename Index, typename Param>
auto close_plane_recovery_births(
    const tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    const tf::buffer<Index> &statement_offsets,
    const tf::buffer<plane_recovery_proposal<Index, Param>> &proposals,
    const tf::buffer<Index> &proposal_offsets,
    tf::buffer<Index> &class_birth) -> bool {
  const auto n_classes = statement_offsets.size() == 0
                             ? std::size_t(0)
                             : statement_offsets.size() - 1;
  class_birth.allocate(n_classes);
  if (n_classes == 0)
    return proposals.size() == 0 && proposal_offsets.size() <= 1;
  if (proposal_offsets.size() != statement_offsets.size())
    return false;

  // a class states its own birth, so every class answers independently and an
  // unnamed birth is the whole refusal
  tf::parallel_for_each(
      tf::make_sequence_range(n_classes),
      [&](std::size_t cls) {
        const auto &name = statements[std::size_t(statement_offsets[cls])].name;
        const auto begin = std::size_t(proposal_offsets[cls]);
        const auto end = std::size_t(proposal_offsets[cls + 1]);
        class_birth[cls] = Index(-1);
        if (begin >= end || proposals[begin].name != name)
          return;
        Index birth = Index(-1);
        for (auto at = begin; at < end; ++at) {
          const auto &candidate = proposals[at];
          if (candidate.name != name || candidate.root == Index(-1) ||
              candidate.plane == Index(-1) ||
              (candidate.tier != Index(0) && candidate.tier != Index(1)) ||
              (candidate.birth != std::uint8_t(0) &&
               candidate.birth != std::uint8_t(1)))
            return;
          if (candidate.birth != std::uint8_t(0) && birth == Index(-1))
            birth = Index(at);
        }
        class_birth[cls] = birth;
      },
      tf::checked);
  return !tf::parallel_contains(
      tf::make_range(class_birth),
      [](Index birth) { return birth == Index(-1); }, tf::checked);
}

} // namespace tf::arrangement
