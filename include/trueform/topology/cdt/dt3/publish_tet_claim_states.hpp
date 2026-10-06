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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_state.hpp"
#include <atomic>
#include <cassert>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto publish_tet_claim_states(Owner &owner,
                              const tet_claim_workspace<Index> &work) -> void {
  tf::parallel_for_each(
      tf::make_sequence_range(work.ownership.size()),
      [&](std::size_t i) {
        const auto word = work.ownership[i].load(std::memory_order_relaxed);
        assert(word == tet_unowned || (word & tet_pooled));
        owner._states[i] =
            word != tet_unowned
                ? tet_state::dead
                : (owner._corners[i][3] == Owner::infinite ? tet_state::infinite
                                                           : tet_state::live);
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3
