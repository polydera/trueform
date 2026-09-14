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

#include <cstdint>

namespace tf::exact::door::pool {

/// ONE EVENT ON THE INSTRUMENT, and the one place `TF_POOL_CENSUS` reaches a
/// counter. Nothing below the door reads either census, so without the macro
/// this is an empty statement and the partition carries no tally at all.
///
/// A phase that would have to WORK to state a fact — a scan, a reduce, a
/// clock — guards the work itself instead, because there is nothing here for
/// the compiler to remove once the work has run.
inline auto tally_census(std::int64_t &counter, std::int64_t by = 1) -> void {
#ifdef TF_POOL_CENSUS
  counter += by;
#else
  (void)counter;
  (void)by;
#endif
}

} // namespace tf::exact::door::pool
