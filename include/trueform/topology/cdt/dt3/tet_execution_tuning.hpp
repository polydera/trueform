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

/// Scheduling and storage choices for the tetrahedralization, in one policy
/// so the operating point is explicit and apart from the geometric
/// operations.
struct tet_execution_tuning {
  static constexpr std::size_t parallel_sites = 4096;
  static constexpr std::size_t parallel_append_sites = 256;
  static constexpr std::size_t minimum_workers = 4;
  static constexpr std::size_t serial_prefix = 128;
  static constexpr std::size_t streams_per_worker = 4;
  static constexpr std::size_t band_growth = 8;
  /// Rows per site a build reserves up front, touching none of them. A
  /// Delaunay complex holds about 6.8 cells per site in general position,
  /// and the lanes grow past the reserve where it holds more.
  static constexpr std::size_t reserve_per_site = 8;
  /// A round of a band that leaves no more sites than this, or inserts none,
  /// is followed by a round that runs the streams one after another.
  static constexpr std::size_t serial_round_sites = 64;
  /// A stream asks for the complex's measured rows per site for every site
  /// it still holds, times the margin, plus the floor; a band takes more rows
  /// per site than the density measured before it, and the barriers inside
  /// the band absorb the difference by measuring again.
  static constexpr double asked_rows_margin = 1.125;
  static constexpr std::size_t asked_rows_floor = 128;
  /// The rows one block of a blocked count-and-write pass scans, and the
  /// block count below which such a pass runs serially.
  static constexpr std::size_t scan_block_rows = 4096;

  /// The turns one walk of a segment may take before the row refuses as
  /// `passes` (@ref recover_tet_edge_by_flips, @ref tet_segment_status):
  /// each turn removes one obstruction between the ends and reads the path
  /// again, so the turns a lawful walk takes are bounded by the cells along
  /// its segment, and 21017 rows is the most any task of the corpus was
  /// measured to touch. The cap states only that the walk is finite.
  static constexpr std::size_t recovery_walk_passes = 65536;

  /// The passes one face's chain may take before the face refuses as
  /// `capped` (@ref recover_tet_face_phase): a pass either reads the cover
  /// again, flips a piercing edge away or places a point, and the longest
  /// chain measured on the corpus is 446 passes, so this is past every chain
  /// measured and states only that the chain is finite.
  static constexpr std::size_t recovery_face_passes = 4096;

  /// The rounds in a row that leave the segment fixpoint no fewer rows
  /// missing before it stops raising the flips' link level and states the
  /// level it tries every star at, and that level
  /// (@ref recover_tet_segments, @ref tet_flip_level_limit). The level caps a
  /// flip search's depth, and this one is past the depth any star of a
  /// complex reaches, so the rounds that follow are bounded by the stars
  /// themselves and by the star limit the caller states, not by the level.
  static constexpr int recovery_stall_rounds = 3;
  static constexpr int recovery_full_level = 100000;

  /// THE GRAIN: the triangles a recovery region holds. A surface of fewer
  /// than twice this is one region, so a small mesh partitions into nothing
  /// and pays nothing at all. The admission measured the frontier at
  /// 16384 / 65536 / 262144 — lucy_x16 21.3 / 11.9 / 6.0 %, igea_x16
  /// 16.5 / 9.6 / 5.2 %, fertility_x64 30.6 / 22.5 / 13.1 % — and the
  /// region's ms followed it (lucy -37.4 / -44.9 / -46.7 % at sixteen
  /// workers), so the coarsest is the grain. `0` partitions nothing, which
  /// is the serial kernel.
  static constexpr std::size_t region_triangles = 262144;
  /// THE ROUND ENGAGES BY SIZE: the due items a round must hold before its
  /// regions are worth dividing, below which the phase runs its serial path
  /// (@ref run_tet_region_round). It is a function of the input alone — the
  /// due count and the partition — so one worker and sixteen run the same
  /// rounds and give one mesh; the round runs at width one too. A mesh whose
  /// every round is below it states nothing and is the serial kernel's, byte
  /// for byte: at 65536 fertility_x64's rounds never meet it and its region
  /// costs 1.1 % of the serial kernel's, while igea_x16's (564588 tasks) and
  /// lucy_x16's (781200) always do.
  static constexpr std::size_t region_round_rows = 65536;
  static constexpr std::size_t serial_scan_blocks = 16;
  /// The sites a tet mesh may hold per site its entrance states: the one
  /// budget the recovery's splits and face points and the drains' mints all
  /// consume, beyond which a split or face point refuses as `budget` and a
  /// drain's mints refuse the run.
  static constexpr std::size_t site_budget_ratio = 32;
  /// The census's rounds on one drain, convict, collapse and recover the
  /// faces moved, at most this many; a census it cuts short counts the faces
  /// its last round moved as refused collapses.
  static constexpr std::size_t census_rounds = 8;
  /// The live rows phase S records per live row of the complex
  /// (@ref measure_tet_phase_damage) above which it restores its own rows
  /// before phase F reads them, instead of leaving one restoration to follow
  /// the region labeling. A phase that rewrote the complex twice over is where
  /// the one restoration's Lawson cascade costs more than two, and where phase
  /// F's own work on a complex left not Delaunay costs more than the
  /// restoration saves.
  static constexpr double phase_restore_density = 2.0;
  /// The coarse blocks of the smallest corner a parallel canonical rank
  /// scatters its keys into before sorting each block, and the block count
  /// below which the pass that orders them runs serially.
  static constexpr std::size_t rank_coarse_blocks = 2048;
  static constexpr std::size_t serial_rank_blocks = 8;
  /// The rows per coarse block a block of that scatter scans at least, so
  /// its counts stay a small fraction of the rows.
  static constexpr std::size_t rank_rows_per_coarse_block = 16;
  /// The longest run of one smallest corner a canonical rank orders by
  /// counting, in lanes on the stack; a longer run is sorted.
  static constexpr std::size_t rank_run_lanes = 64;
  /// The coarse blocks of the key a grouping of records by key scatters its
  /// records into before each places its own (@ref
  /// scatter_tet_records_by_key), and the sources per coarse block a block
  /// of that scatter reads at least.
  static constexpr std::size_t record_coarse_keys = 2048;
  static constexpr std::size_t sources_per_coarse_key = 16;
  /// The sources plus keys below which that grouping takes its one-lane
  /// kernel at any worker count: below it the coarse blocks' fixed cost
  /// outweighs the work they divide.
  static constexpr std::size_t record_serial_work = std::size_t(1) << 19;
  /// The bits of the lower corner one counting pass of an overflowed
  /// boundary's edge records orders by, and the longest run of one lower
  /// corner whose mates are found by scanning it; a longer run is sorted by
  /// its upper corner.
  static constexpr std::size_t cavity_record_bits = 11;
  static constexpr std::size_t cavity_run_scan = 16;
};

} // namespace tf::topology::cdt::dt3
