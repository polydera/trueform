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

#include "obj_execution_tuning.hpp"

#include <tbb/task_arena.h>

#include <algorithm>
#include <cstddef>

namespace tf::io::obj {

inline auto obj_partition_count(std::size_t size) -> std::size_t {
  const auto available = std::max(
      std::size_t(1),
      static_cast<std::size_t>(tbb::this_task_arena::max_concurrency()) *
          obj_execution_tuning::partitions_per_worker);
  const auto useful =
      size == 0 ? std::size_t(1)
                : (size - 1) / obj_execution_tuning::target_partition_bytes + 1;
  return std::min(available, useful);
}

} // namespace tf::io::obj
