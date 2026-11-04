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
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Scheduling and storage choices for the tetrahedralization. Keeping them in
/// one policy makes the operating point explicit and lets benchmark targets
/// sweep it without changing the geometric operations.
struct tet_execution_tuning {
  static constexpr std::size_t parallel_sites = 4096;
  static constexpr std::size_t parallel_append_sites = 256;
  static constexpr std::size_t minimum_workers = 4;
  static constexpr std::size_t serial_prefix = 128;
  static constexpr std::size_t streams_per_worker = 4;
  static constexpr std::size_t band_growth = 8;
  /// Rows per site, pooled for a streamed site and reserved up front by a
  /// serial build. A Delaunay complex holds about 6.8 cells per site in
  /// general position; a stream that still runs dry takes rows from
  /// finished streams at the barrier, and what no stream can take is left
  /// to the serial tail, which grows the lanes.
  static constexpr std::size_t reserve_per_site = 8;
  /// A blocked stream asks the barrier for this many rows per site it still
  /// holds, plus the floor.
  static constexpr std::size_t borrowed_rows_per_site = 16;
  static constexpr std::size_t borrowed_rows_floor = 128;
  /// The rows one block of a blocked count-and-write pass scans, and the
  /// block count below which such a pass runs serially.
  static constexpr std::size_t scan_block_rows = 4096;
  static constexpr std::size_t serial_scan_blocks = 16;
  /// The sites a tet mesh may hold per site its entrance states, beyond
  /// which the splits that would pass them refuse as `budget`. On 199
  /// closed Thingi10K models at 8 workers a ratio of 16 meshes 80, its
  /// refusals taking 0.1 s at the median and 30 s at most; 32 meshes 115,
  /// losing none, its refusals 0.5 s at the median and 83 s at most; 64
  /// meshed 6 of 16 sampled refusals against 32's 3, at refusals up to
  /// 257 s.
  static constexpr std::size_t site_budget_ratio = 32;
  /// The coarse blocks of the smallest corner a parallel canonical rank
  /// scatters its keys into before sorting each block.
  static constexpr std::size_t rank_coarse_blocks = 2048;
};

} // namespace tf::topology::cdt::dt3
