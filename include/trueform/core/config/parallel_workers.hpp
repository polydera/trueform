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
#include <algorithm>
#include <cstddef>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

namespace tf::core {

/// The workers the current arena will run: its concurrency, bounded by the
/// active global limit on parallelism. Every guard that admits a parallel
/// scheme or sizes a partition per worker reads this and nothing else.
inline auto parallel_workers() -> std::size_t {
  return std::min(std::size_t(tbb::this_task_arena::max_concurrency()),
                  tbb::global_control::active_value(
                      tbb::global_control::max_allowed_parallelism));
}

} // namespace tf::core
