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
  /// Pool rows per streamed site. A Delaunay complex holds about 6.8 cells
  /// per site in general position; a stream that still runs dry takes rows
  /// from finished streams at the barrier, and what no stream can take is
  /// left to the serial tail, which grows the lanes.
  static constexpr std::size_t reserve_per_site = 8;
  /// A blocked stream asks the barrier for this many rows per site it still
  /// holds, plus the floor.
  static constexpr std::size_t borrowed_rows_per_site = 16;
  static constexpr std::size_t borrowed_rows_floor = 128;
  /// The rows one block of a blocked count-and-write pass scans, and the
  /// block count below which such a pass runs serially.
  static constexpr std::size_t scan_block_rows = 4096;
  static constexpr std::size_t serial_scan_blocks = 16;
  static constexpr std::size_t comparison_order_sites = 768;
  static constexpr unsigned order_radix_bits = 11;
};

} // namespace tf::topology::cdt::dt3
