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
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Grow the protection lane and the owner lane to every row, the new rows
/// unprotected and unowned, so writers sharing the rows only state bits and
/// owners. The new rows are the carrier and are stated in parallel, a round
/// reserving a pool bringing in as many at once as the complex holds; a
/// growth of a few stays serial.
template <typename Owner> auto reserve_tet_protection(Owner &owner) -> void {
  auto &slots = owner._protected_slots;
  auto &owners = owner._owners;
  const std::size_t rows = std::size_t(owner._corners.size());
  if (slots.size() >= rows)
    return;
  const std::size_t old = slots.size();
  slots.reallocate(rows);
  owners.reallocate(4 * rows);
  tf::parallel_for_each(
      tf::make_sequence_range(old, rows),
      [&slots, &owners](std::size_t row) {
        slots[row] = 0;
        for (std::size_t slot = 0; slot < 4; ++slot)
          owners[4 * row + slot] = Owner::none;
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3
