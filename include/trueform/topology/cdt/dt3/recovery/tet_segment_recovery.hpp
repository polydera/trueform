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
#include "../tet_flip_scratch.hpp"
#include "../tet_site_candidate.hpp"
#include "./tet_chain_vertex.hpp"
#include "./tet_segment_end.hpp"
#include "./tet_segment_status.hpp"
#include "./tet_star_stamps.hpp"
#include "./tet_stuck_facet.hpp"
#include "./tet_subsegment.hpp"
#include "./tet_subsegment_query.hpp"
#include "./tet_subsegment_split.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

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
/// `chains` holds per segment its vertices from its first end to its last;
/// `status` and `rounds` say what became of it and in which round;
/// `shell_splits` counts the splits its corner shells proposed; `rings`
/// holds per segment end, `2 * segment + end`, the numerator its corner's
/// shell placed and no split has realized yet, `-1` for none: a row at the
/// corner that stood when its turn came deferred it, and splits there once
/// it is read missing again;
/// `flips` is the scratch the flip removals walk on, `flip_recovered` and
/// `search_recovered` count the rows flips recovered without and with the
/// full search, `split_rows` the rows that split;
/// `site_budget` bounds the sites the complex may hold, the splits that
/// would pass it refused as `budget`; `placed` holds
/// every site the call placed with a cell row near it, in placement order, the
/// last round's from `placed_from`. `row_heads` and `row_tickets` chain each
/// site to the rows ticketed at it (@ref ticket_tet_subsegment), and `revisits`
/// lists the rows the next round reads for their own sake: split, withdrawn,
/// declined or deferred this round. `queue` holds the rows a round reads,
/// `deferred` the ones its flips left standing, the next round's queue.
/// `declined` marks the fresh splits of the last append whose rows stood when
/// their turn came. `order` holds the rows by segment and first end, which the
/// chains are walked through; `round` counts the rounds run; `tickets` is the
/// restoration's facet queue
/// (@ref restore_tet_delaunay), `stuck` and `stuck_next` the facets its
/// Lawson pass could not flip, `tried` per row a bit per edge slot
/// (@ref tet_edge_slot), set when a removal of that edge failed with the row
/// in its ring and cleared when the owner records a write of the row — a
/// ring's hull cells follow from its finite ones — and `unrestored` the
/// edges it left not locally Delaunay.
///
/// Phase F reads pieces again in the constrained complex
/// (@ref reenter_tet_segments), only those an insertion touched:
/// `redirected`
/// holds, sorted, the rows to split however they stand, each with the end
/// it had, so a row keeps its redirect until a split lands, and `revived`
/// the recovered segments read again.
template <typename Index, typename Coord> struct tet_segment_recovery {
  tf::offset_block_buffer<Index, tet_chain_vertex<Index>> chains;
  tf::buffer<tet_segment_status> status;
  tf::buffer<Index> rounds;

  tf::buffer<tet_subsegment<Index>> subsegments;
  tf::buffer<int> bits;
  tf::buffer<Index> first;
  tf::buffer<Index> last;
  tf::buffer<tet_segment_end<Index>> ends;
  tf::buffer<char> acute;
  tf::buffer<char> shelled;
  tf::buffer<std::int64_t> rings;
  tf::buffer<Index> dirty;
  tf::buffer<Index> queue;
  tf::buffer<char> queued;
  tf::buffer<char> fresh;
  tf::buffer<std::size_t> corners;
  tf::buffer<Index> refused;
  tf::buffer<Index> deferred;
  tf::buffer<Index> revisits;
  tf::buffer<Index> order;
  tf::buffer<std::array<Index, 2>> redirected;
  tf::buffer<Index> revived;
  tf::buffer<Index> row_heads;
  tf::buffer<std::pair<Index, Index>> row_tickets;
  tf::buffer<Index> proposed;
  tf::buffer<tet_subsegment_query<Index>> queries;
  tf::buffer<tet_subsegment_split<Index, Coord>> splits;
  tf::buffer<tet_subsegment_split<Index, Coord>> statement;
  tf::buffer<Index> targets;
  tf::buffer<char> declined;
  tf::buffer<Index> star;
  tf::buffer<Index> tickets;
  tf::buffer<std::array<Index, 2>> placed;
  tf::buffer<Index> minted;
  tf::buffer<Index> touched;
  tf::buffer<std::uint32_t> stamps;
  tf::buffer<tet_site_candidate<Coord>> candidates;
  tf::offset_block_buffer<Index, Index> site_cells;
  tet_star_stamps star_stamps;
  tet_flip_scratch<Index> flips;
  tf::buffer<tet_stuck_facet<Index>> stuck;
  tf::buffer<tet_stuck_facet<Index>> stuck_next;
  tf::buffer<std::uint8_t> tried;
  std::size_t flip_recovered = 0;
  std::size_t search_recovered = 0;
  std::size_t split_rows = 0;
  std::size_t shell_splits = 0;
  std::size_t unrestored = 0;
  std::size_t first_minted = 0;
  std::size_t placed_from = 0;
  std::size_t site_budget = std::numeric_limits<std::size_t>::max();
  bool deepens = false;
  std::uint32_t stamp = 0;
  Index round = 0;
};

} // namespace tf::topology::cdt::dt3
