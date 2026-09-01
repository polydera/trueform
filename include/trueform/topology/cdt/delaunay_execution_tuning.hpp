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

namespace tf::topology::cdt {

/// Scheduling and storage choices for the Delaunay pipeline. Keeping them in
/// one policy makes the measured operating point explicit and lets benchmark
/// targets sweep it without changing the geometric operations.
struct delaunay_execution_tuning {
  static constexpr std::size_t leaf_sites = 16;
  static constexpr std::size_t comparison_order_sites = 768;
  static constexpr std::size_t parallel_load_sites = 50000;
  static constexpr std::size_t parallel_order_sites = 100000;
  static constexpr std::size_t parallel_topology_sites = 50000;
  static constexpr std::size_t parallel_task_sites = 16384;
  static constexpr std::size_t tasks_per_worker = 4;
  static constexpr std::size_t edge_arena_block_darts = 4096;
  static constexpr std::size_t reserved_darts_per_site = 9;
  static constexpr std::size_t parallel_face_darts = 100000;
  static constexpr std::size_t face_chunk_darts = 65536;
};

} // namespace tf::topology::cdt
