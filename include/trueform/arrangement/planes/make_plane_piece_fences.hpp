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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "./make_plane_arrangement_cells.hpp"
#include "./make_plane_piece_incidence.hpp"
#include "./plane_arrangement.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// Do the cells this piece bounds carry different depths?
///
/// The rows arrive one per member covering the cell, so a cell's run length
/// IS its depth and a border between depths is a run length change.
template <typename Index>
auto plane_piece_depth_varies(tf::buffer<Index> &cells) -> bool {
  if (cells.size() == 0)
    return false;
  std::sort(cells.begin(), cells.end());
  std::size_t reference = 1;
  while (reference < cells.size() && cells[reference] == cells[0])
    ++reference;
  for (std::size_t at = reference; at < cells.size();) {
    std::size_t end = at + 1;
    while (end < cells.size() && cells[end] == cells[at])
      ++end;
    if (end - at != reference)
      return true;
    at = end;
  }
  return false;
}

/// One verdict pair per final piece: `fan` for the radial tier,
/// `crossable` for the component flood.
struct plane_piece_fences {
  tf::buffer<char> fan;
  tf::buffer<char> crossable;
};

/// A piece fences for exactly three reasons, and nothing else:
///
/// FAN — planes meet here and a radial pairing stands. TWO independent facts
/// say so, and this tier is where they are read together: an instance states
/// the SEAM (@ref tf::intersect::graph::plane_edge_fan_flag), or MORE THAN
/// TWO live cell incidences meet at the piece. The seam is an input fact like
/// the non-manifold bit — a second operand touches here, which at incidence
/// two no count can see, and its two pages divide trivially. The incidence is
/// the arrangement's own: past two the radial order has something to decide,
/// whatever named the piece. It counts OCCURRENCES — a cell reaching the
/// piece from both sides states two of them, and a coplanar duplicate states
/// none, its survivor already carrying their shared cell, so a stack's depth
/// never inflates the count.
///
/// The other two fence quietly — no fan stands there. The mesh's own
/// NON-MANIFOLD edge, an input fact the form's edge link states and the
/// instances carry (@ref tf::intersect::graph::plane_edge_non_manifold_flag),
/// where no single continuation exists. A border between DEPTHS: the number
/// of sheets covering the ground changes across the piece — the one fact the
/// live count cannot carry, since it is the dead members that differ. Equal
/// depth is ground continuing, and continuing ground is CROSSABLE — what the
/// component flood may join. A piece no cell names is dead and fences
/// nothing.
template <typename Index, typename Int, typename Immutable>
auto make_plane_piece_fences(const plane_arrangement<Index, Int> &arrangement,
                             const Immutable &immutable,
                             const plane_piece_incidence<Index> &incidence,
                             const plane_arrangement_cells<Index> &cells)
    -> plane_piece_fences {
  plane_piece_fences fences;
  // every piece of the ticket space gets both verdicts on every path out of
  // the callback, so the uninitialized allocation is total when it returns
  const auto n_pieces = std::size_t(arrangement.final_piece_ticket_extent());
  fences.fan.allocate(n_pieces);
  fences.crossable.allocate(n_pieces);
  const auto coplanar_of = arrangement.coplanar_of();
  struct local_t {
    tf::buffer<Index> depth_cells;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(arrangement.final_piece_ticket_extent()),
      [&](Index piece, local_t &local) {
        const auto rows = incidence.rows_of_piece[std::size_t(piece)];
        if (rows.size() == 0) {
          fences.fan[std::size_t(piece)] = char(0);
          fences.crossable[std::size_t(piece)] = char(1);
          return;
        }
        // one row per live occurrence is the degree; one triangle naming
        // this piece from two slots is still ONE covering member, and the
        // incidence fills a piece's rows triangle-ascending, so the depth's
        // repeat is the previous row's triangle
        local.depth_cells.clear();
        Index live = 0;
        Index previous = Index(-1);
        for (const auto row : rows) {
          const auto triangle = row / Index(3);
          live +=
              Index(coplanar_of[std::size_t(triangle)] == Index(-1));
          if (triangle == previous)
            continue;
          previous = triangle;
          local.depth_cells.push_back(cells.cell_of[std::size_t(triangle)]);
        }
        if (live > Index(2)) {
          fences.fan[std::size_t(piece)] = char(1);
          fences.crossable[std::size_t(piece)] = char(0);
          return;
        }
        bool non_manifold = false;
        for (const auto &definition :
             arrangement.piece_definitions(immutable, piece)) {
          if ((definition.flags &
               tf::intersect::graph::plane_edge_fan_flag) != 0) {
            fences.fan[std::size_t(piece)] = char(1);
            fences.crossable[std::size_t(piece)] = char(0);
            return;
          }
          non_manifold =
              non_manifold ||
              (definition.flags &
               tf::intersect::graph::plane_edge_non_manifold_flag) != 0;
        }
        fences.fan[std::size_t(piece)] = char(0);
        if (non_manifold) {
          fences.crossable[std::size_t(piece)] = char(0);
          return;
        }
        fences.crossable[std::size_t(piece)] =
            char(!plane_piece_depth_varies(local.depth_cells));
      },
      local_t{}, tf::checked);
  return fences;
}

} // namespace tf::arrangement
