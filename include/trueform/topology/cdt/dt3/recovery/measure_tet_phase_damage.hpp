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
#include "../../../../core/algorithm/block_reduce.hpp"
#include "../../../../core/checked.hpp"
#include "../tet_state.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// How densely a phase's flips and splits left the complex `owner` damaged:
/// the live rows it recorded per live row the complex holds. The recorded
/// rows are the ones a restoration would seed from, so the density is the
/// cascade one restoration faces, read before it runs. Counted over the
/// record's writes, so a row written many times weighs as many times; the
/// count is an integer sum over a fixed set and the division follows it, so
/// the density is one number at any worker count.
template <typename Owner>
auto measure_tet_phase_damage(const Owner &owner) -> double {
  const std::size_t rows = owner._corners.size() - owner._n_dead;
  if (!rows)
    return 0;
  struct local_t {
    std::size_t recorded = 0;
  };
  std::size_t recorded = 0;
  tf::blocked_reduce(
      owner._written, recorded, local_t{},
      [&owner](const auto &block, local_t &local) {
        for (const auto &record : block)
          if (owner._states[std::size_t(record.row)] == tet_state::live)
            local.recorded += record.writes;
      },
      [](const local_t &local, std::size_t &total) { total += local.recorded; },
      tf::checked);
  return double(recorded) / double(rows);
}

} // namespace tf::topology::cdt::dt3
