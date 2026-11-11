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
#include "../../../core/buffer.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The pairing of a cavity boundary's facets across their shared edges: the
/// three side neighbours of every cell the star writes. A boundary vertex
/// takes a local id at its first touch: its byte in `local_of_site` names the
/// id, and the id is its own while `site_of_local` names the vertex back below
/// `n_local`, so a byte some earlier boundary left is never read as this one's
/// and no lane is cleared; the three ids past the last local are a facet's
/// overflow. An edge indexes `table` by its two local ids, the
/// slot holding the first facet-edge's ticket until the second arrives, pairs
/// them in `across` and empties the slot. Every edge of a closed boundary is
/// visited exactly twice, so the table is empty between insertions and nothing
/// is cleared on the way; a boundary a flood abandons is scrubbed at the next
/// open. A boundary of more than `max_local` vertices leaves the table
/// (`overflow_tet_cavity_links`) and is paired at the close from its edge
/// records, ordered by their lower corner (`order_tet_cavity_records`) with
/// every record's mate beside it (`close_tet_cavity_links`); the run of one
/// lower corner holds no order of its own, so an edge is answered by scanning
/// its run (`tet_cavity_links_hold`). A
/// corner addresses its stamp at `corner - lowest_corner`, so the owner's
/// sentinel corners stamp like sites. `n_facets` is the fill of `facets` and
/// `across` (three per facet); their size is grown capacity.
template <typename Index> struct tet_cavity_links {
  static constexpr std::size_t max_local = 256;
  static constexpr Index lowest_corner = -2;
  static constexpr Index marked = -2;
  tf::buffer<std::array<Index, 3>> facets;
  tf::buffer<Index> across;
  std::size_t n_facets = 0;
  tf::buffer<std::uint8_t> local_of_site;
  std::array<Index, max_local + 3> site_of_local{};
  tf::buffer<Index> table;
  tf::buffer<std::array<Index, 3>> records;
  tf::buffer<std::array<Index, 3>> record_scratch;
  tf::buffer<std::uint32_t> record_counts;
  std::size_t n_local = 0;
  bool overflowed = false;
  bool pending = false;
  Index sink = -1;
};

template <typename Index>
auto tet_cavity_stamp_slot(Index corner) -> std::size_t {
  return std::size_t(corner - tet_cavity_links<Index>::lowest_corner);
}

} // namespace tf::topology::cdt::dt3
