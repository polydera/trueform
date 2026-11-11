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
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/cdt/dt3/tet_facet_is_protected.hpp"
#include "../../topology/cdt/dt3/tet_facet_key.hpp"
#include "../../topology/cdt/dt3/tet_facet_owner.hpp"
#include "../../topology/cdt/dt3/tet_shared_owners.hpp"
#include "../../topology/cdt/dt3/tet_state.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::tetmesh {

/// Into `records`, one `{key, face}` per face owning a protected facet of
/// the live complex, read off the owner lane and a shared facet's stack:
/// each facet once, from the lower of its two rows, in row order.
template <typename Owner>
auto gather_tet_face_owners(const Owner &owner,
                            tf::buffer<tf::topology::cdt::dt3::tet_facet_owner<
                                typename Owner::index_type>> &records) -> void {
  using Index = typename Owner::index_type;
  namespace dt3 = tf::topology::cdt::dt3;
  const std::size_t rows = std::min(std::size_t(owner._corners.size()),
                                    owner._protected_slots.size());
  records.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(rows), records,
      [&owner](std::size_t cell, auto &out) {
        if (owner._states[cell] == dt3::tet_state::dead ||
            !owner._protected_slots[cell])
          return;
        for (std::size_t slot = 0; slot < 4; ++slot) {
          if (!dt3::tet_facet_is_protected(owner, Index(cell), slot) ||
              std::size_t(owner._neighbors[cell][slot]) < cell)
            continue;
          const Index who = owner._owners[4 * cell + slot];
          if (who == Owner::none)
            continue;
          const auto key = dt3::tet_facet_key(owner._corners[cell], slot);
          if (who != Owner::shared_owner) {
            out.push_back({key, who});
            continue;
          }
          const auto stack = dt3::tet_shared_owners(owner, key);
          for (auto at = stack.first; at != stack.second; ++at)
            out.push_back({key, at->face});
        }
      },
      tf::checked);
}

} // namespace tf::tetmesh
