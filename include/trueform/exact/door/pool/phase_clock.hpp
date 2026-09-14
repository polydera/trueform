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

#ifdef TF_POOL_CENSUS
#include <chrono>
#endif

namespace tf::exact::door::pool {

/// THE INSTRUMENT'S CLOCK: seconds since the last phase boundary, and the
/// boundary moved to here.
///
/// Without `TF_POOL_CENSUS` a phase reads zero and `<chrono>` never enters
/// the door's include graph — which every translation unit that builds an
/// arrangement carries.
struct phase_clock {
#ifdef TF_POOL_CENSUS
  std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now();

  auto since() -> double {
    const auto now = std::chrono::steady_clock::now();
    const auto spent = std::chrono::duration<double>(now - at).count();
    at = now;
    return spent;
  }
#else
  auto since() -> double { return 0.0; }
#endif
};

} // namespace tf::exact::door::pool
