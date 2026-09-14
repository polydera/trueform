/*
 * Copyright (c) 2025 XLAB
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

#include "trueform/core/checked.hpp"
#include "trueform/cpp/core/parallel_config.hpp"

#include <cstddef>
#include <limits>

namespace tf::cpp {

/// @brief `tf::checked` for a carrier costing the product of `factors`
/// elementwise operations.
///
/// The primitive compares the carrier count, so the tier's work threshold
/// becomes that count divided by what one carrier costs. The quotient is taken
/// factor by factor, which never overflows and equals the quotient by their
/// product. A carrier costing nothing keeps the whole range serial.
template <typename... Factors>
constexpr auto checked_work(Factors... factors) -> tf::checked_t {
  const std::size_t costs[] = {static_cast<std::size_t>(factors)..., 1};
  auto carriers = parallel_threshold;
  for (const auto cost : costs) {
    if (cost == 0)
      return {std::numeric_limits<unsigned long>::max()};
    carriers = carriers / cost + (carriers % cost != 0);
  }
  return {static_cast<unsigned long>(carriers)};
}

static_assert(checked_work(1).serial_below == parallel_threshold);
static_assert(checked_work().serial_below == parallel_threshold);
static_assert(checked_work(3).serial_below == (parallel_threshold + 2) / 3);
static_assert(checked_work(2, 3).serial_below == (parallel_threshold + 5) / 6);
static_assert(checked_work(parallel_threshold).serial_below == 1);
static_assert(
    checked_work(std::numeric_limits<std::size_t>::max()).serial_below == 1);
static_assert(checked_work(std::numeric_limits<std::size_t>::max(),
                           std::numeric_limits<std::size_t>::max())
                  .serial_below == 1);
static_assert(checked_work(0).serial_below ==
              std::numeric_limits<unsigned long>::max());
static_assert(checked_work(std::numeric_limits<std::size_t>::max(), 0)
                  .serial_below == std::numeric_limits<unsigned long>::max());

} // namespace tf::cpp
