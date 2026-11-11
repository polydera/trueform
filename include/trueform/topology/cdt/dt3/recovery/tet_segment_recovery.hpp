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
#include "../../../../core/buffer.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "../tet_edge_star.hpp"
#include "../tet_flip_scratch.hpp"
#include "../tet_site_candidate.hpp"
#include "./tet_crossing_due.hpp"
#include "./tet_row_link.hpp"
#include "./tet_segment_end.hpp"
#include "./tet_segment_status.hpp"
#include "./tet_site_cell_scratch.hpp"
#include "./tet_star_stamps.hpp"
#include "./regions/tet_round_workspace.hpp"
#include "./tet_stuck_facet.hpp"
#include "./tet_subsegment.hpp"
#include "./tet_subsegment_query.hpp"
#include "./tet_subsegment_split.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tf::topology::cdt::dt3 {

/// Segment recovery's product and the scratch its rounds reuse.
///
/// A protected segment is an edge constraint, and an edge is no wall: a cavity
/// reaches around it, so the flood cannot keep one. A segment is kept by
/// splitting it until every piece is an edge of the complex — at its dyadic
/// midpoints, at a site standing on it, which becomes a shared vertex, and at
/// an acute corner on a shell of one radius, taken only where a midpoint split
/// already failed at that corner. The pieces are therefore this operation's
/// records, not the owner's, and a later insertion that encroaches one is its
/// supplier's to split.
///
/// A segment's frame is 2^bits, bits the depth or fewer where its ends are
/// not that finely scaled, so every split point is a lattice point on the
/// segment; where `deepens`, a segment whose frame has no split left
/// deepens it a bit at a time (@ref deepen_tet_segment_frame), its splits
/// standing exactly on it below the frame.
///
/// A segment's chain is its rows, `first` through each row's `next` to
/// `last`, and nothing else (@ref for_each_tet_chain_piece);
/// `status` and `rounds` say what became of it and in which round;
/// `shell_splits` counts the splits its corner shells proposed; `rings`
/// holds per segment end, `2 * segment + end`, the numerator its corner's
/// shell placed and no split has realized yet, `-1` for none: a row at the
/// corner that stood when its turn came deferred it, and splits there once
/// it is read missing again;
/// `flips` is the scratch the flip removals walk on, `flip_recovered` and
/// `search_recovered` count the rows flips recovered without and with the
/// full search, `split_rows` the rows that split, `crossing_rows` the rows
/// split at a crossing;
/// `site_budget` bounds the sites the complex may hold, the splits that
/// would pass it refused as `budget`; `reach` is the input quantum on the
/// frame, the side of the grid whose cell a point the recovery would mint
/// shares with a corner names that corner instead (@ref tet_shares_reach_cell),
/// `0` naming only a site standing at it, and `steiner` per slot whether the
/// recovery minted the site, the slots past it none, so a site it names `0` is
/// a corner of the input
/// (@ref tet_site_is_corner); `minted` holds per site the call
/// minted, from `first_minted` on, the segment whose split placed it.
/// `row_heads` holds per site the first of
/// the rows whose lower site it is and `row_links` per row its place among
/// them (@ref ticket_tet_subsegment), so a piece's row is read among the
/// rows standing at its lower site alone (@ref tet_piece_row); they live the
/// build, and a retirement withdraws its segment's rows, so a chain holds
/// only the rows of segments still a constraint; `revisits`
/// lists the rows the next round reads for their own sake: split, withdrawn,
/// declined or deferred this round. THE REGIONS: `site_patches` is the patch
/// of every site — THE ONE TABLE THE ADMISSION READS, a row's own region
/// being its first corner's patch (@ref tet_row_admits) — appended to as the
/// frontier mints, and `n_regions` is the regions the partition named at the
/// first round that engaged
/// (@ref tet_execution_tuning::region_triangles the grain), `round_work` the workspace one
/// round of them walks on and `region_left` the positions the admission
/// stopped, which the round runs itself in queue order
/// (@ref run_tet_region_round), `region_left_records` the restoration's own
/// records the admission stopped, `region_seeds` the restoration's rows one
/// sweep divides; `region_tasks` and `region_deferred` count
/// them across the phase, the frontier's share being their ratio.
/// `queue` holds the rows a round reads,
/// `deferred` the ones its flips left standing, the next round's queue.
/// `declined` marks the fresh splits of the last append whose rows stood when
/// their turn came. `round` counts the rounds run, across every statement;
/// the restoration's facet queue and the facets its sweep could not flip are
/// the task's own (@ref tet_flip_scratch), `stuck_next` the records one round
/// hands the next, `tried` per row a
/// bit per edge slot
/// (@ref tet_edge_slot), set when a removal of that edge failed with the row
/// in its ring and cleared when the owner records a write of the row, a
/// ring's hull cells following from its finite ones; it lives one
/// restoration, which empties it as it leaves. `passes_capped` counts the
/// rows whose walk spent the turns a walk may take, each refusing its
/// segment as `passes` (@ref recover_tet_edge_by_flips). `unrestored` counts the
/// edges a restoration left not locally Delaunay, and
/// `restored_before_faces` states that phase S restored its own rows rather
/// than leave them to the one restoration after the region labeling
/// (@ref recover_tet_segment_phase).
///
/// `due` holds the constraints a cut made due (@ref tet_crossing_due).
/// `collapse_link` and `collapse_rows` are the lists a collapse of a site
/// walks on, its link and the rows naming it (@ref collapse_tet_site_rows).
template <typename Index, typename Coord> struct tet_segment_recovery {
  tf::buffer<tet_segment_status> status;
  tf::buffer<Index> rounds;

  tf::buffer<tet_subsegment<Index>> subsegments;
  tf::buffer<int> bits;
  tf::buffer<Index> first;
  tf::buffer<Index> last;
  tf::buffer<tet_segment_end<Index>> ends;
  /// The tables the ends are bucketed on: the ends staged on their way,
  /// their runs by site and the blocked counts (@ref bucket_tet_records).
  tf::buffer<tet_segment_end<Index>> end_scratch;
  tf::buffer<Index> end_runs;
  tf::buffer<Index> bucket_counts;
  tf::buffer<char> acute;
  tf::buffer<char> shelled;
  tf::buffer<std::int64_t> rings;
  tf::buffer<Index> dirty;
  std::size_t n_regions = 0;
  tf::buffer<Index> site_patches;
  tet_round_workspace<Index> round_work;
  tf::buffer<Index> region_left;
  tf::buffer<tet_stuck_facet<Index>> region_left_records;
  tf::buffer<Index> region_seeds;
  std::size_t region_tasks = 0;
  std::size_t region_deferred = 0;
  tf::buffer<Index> queue;
  tf::buffer<char> queued;
  tf::buffer<char> fresh;
  tf::buffer<std::size_t> corners;
  tf::buffer<Index> refused;
  tf::buffer<Index> deferred;
  tf::buffer<Index> revisits;
  tf::buffer<Index> row_heads;
  tf::buffer<tet_row_link<Index>> row_links;
  tf::buffer<Index> proposed;
  tf::buffer<tet_subsegment_query<Index>> queries;
  tf::buffer<tet_subsegment_split<Index, Coord>> splits;
  tf::buffer<tet_subsegment_split<Index, Coord>> statement;
  tf::buffer<Index> targets;
  tf::buffer<char> declined;
  tf::buffer<Index> minted;
  tf::buffer<Index> touched;
  tf::buffer<tet_site_candidate<Coord>> candidates;
  tf::offset_block_buffer<Index, Index> site_cells;
  tf::buffer<Index> star_sites;
  tet_site_cell_scratch<Index> site_scratch;
  tet_flip_scratch<Index> flips;
  tf::buffer<tet_stuck_facet<Index>> stuck_next;
  tet_crossing_due<Index> due;
  tf::buffer<std::uint8_t> tried;
  tf::buffer<Index> collapse_link;
  tf::buffer<Index> collapse_rows;
  std::size_t flip_recovered = 0;
  std::size_t search_recovered = 0;
  std::size_t split_rows = 0;
  std::size_t crossing_rows = 0;
  std::size_t shell_splits = 0;
  std::size_t unrestored = 0;
  std::size_t passes_capped = 0;
  std::size_t first_minted = 0;
  std::size_t site_budget = std::numeric_limits<std::size_t>::max();
  Coord reach = 0;
  tf::buffer<char> steiner;
  bool deepens = false;
  bool restored_before_faces = false;
  Index round = 0;
};

} // namespace tf::topology::cdt::dt3
