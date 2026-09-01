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

#include <cstddef>

namespace tf::io::obj {

struct obj_execution_tuning {
  static constexpr std::size_t parallel_file_bytes = 256 * 1024;
  static constexpr std::size_t target_partition_bytes = 256 * 1024;
  static constexpr std::size_t partitions_per_worker = 4;
};

} // namespace tf::io::obj
