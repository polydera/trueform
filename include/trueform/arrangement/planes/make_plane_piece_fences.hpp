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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "./make_plane_arrangement_cells.hpp"
#include "./make_plane_piece_incidence.hpp"
#include "./plane_arrangement.hpp"
#include "./plane_arrangement_face.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// What a cell's covering states when crossed: one entry per covering
/// member, its form tag and its winding against the carrier.
template <typename Index> struct plane_piece_crossing_entry {
  Index cell;
  Index tag;
  Index winding;
};

/// Do the cells this piece bounds carry different depths that are not one
/// crossing?
///
/// The rows arrive one per member covering the cell, so a cell's run length
/// IS its depth, and a cell's crossing is the net winding of every tag
/// covering it — a fold's layers cancel, a second form's wall counts once.
/// A border between depths stands only where the crossing changes with it:
/// a pleat two or six sheets deep is the ground it folds out of. Equal
/// depths are ground continuing, whatever their windings, as they always
/// were.
template <typename Index>
auto plane_piece_depth_varies(
    tf::buffer<plane_piece_crossing_entry<Index>> &entries) -> bool {
  if (entries.size() == 0)
    return false;
  std::sort(entries.begin(), entries.end(),
            [](const auto &a, const auto &b) {
              return a.cell != b.cell ? a.cell < b.cell : a.tag < b.tag;
            });
  std::size_t reference_end = 0;
  while (reference_end < entries.size() &&
         entries[reference_end].cell == entries[0].cell)
    ++reference_end;
  const auto next_layer = [&](std::size_t at, std::size_t end, Index &tag,
                              Index &sum) {
    tag = entries[at].tag;
    sum = Index(0);
    while (at < end && entries[at].tag == tag)
      sum += entries[at++].winding;
    return at;
  };
  for (std::size_t at = reference_end; at < entries.size();) {
    std::size_t end = at;
    while (end < entries.size() && entries[end].cell == entries[at].cell)
      ++end;
    if (end - at != reference_end) {
      // the depth changed: a border unless the net layers are the same
      std::size_t r = 0, c = at;
      bool same = true;
      while (same && r < reference_end && c < end) {
        Index rt, rs, ct, cs;
        r = next_layer(r, reference_end, rt, rs);
        c = next_layer(c, end, ct, cs);
        same = rt == ct && rs == cs;
      }
      if (!same || r != reference_end || c != end)
        return true;
    }
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
/// FAN — sheets meet here and a radial pairing stands. THREE independent
/// facts say so, and this tier is where they are read together: an instance
/// states the SEAM (@ref tf::intersect::graph::plane_edge_fan_flag), the
/// instance carries the mesh's own NON-MANIFOLD edge
/// (@ref tf::intersect::graph::plane_edge_non_manifold_flag), or MORE THAN
/// TWO live cell incidences meet at the piece. The seam is an input fact
/// like the non-manifold bit — a second operand touches here, which at
/// incidence two no count can see, and its two pages divide trivially. The
/// non-manifold bit admits the ring even when its sheets are not all cut:
/// the ring gathers the uncut sheets itself, so the live count cannot be
/// the gate. The incidence is the arrangement's own: past two the radial
/// order has something to decide, whatever named the piece. It counts
/// OCCURRENCES — a cell reaching the piece from both sides states two of
/// them, and a coplanar duplicate states none, its survivor already
/// carrying their shared cell, so a stack's depth never inflates the count.
///
/// The last fences quietly — no fan stands there. A border between DEPTHS:
/// the number of sheets covering the ground changes across the piece — the
/// one fact the live count cannot carry, since it is the dead members that
/// differ — and the crossing changes with it, the net winding per form of
/// those sheets. A fold's own layers cancel, so a pleat two or six sheets
/// deep is the same ground as the wall it folds out of and no border. Equal
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
  // the face of every row, so a covering member states its tag and winding
  tf::buffer<Index> face_of_row;
  face_of_row.allocate(arrangement.triangles().size());
  tf::parallel_fill(face_of_row, Index(-1));
  tf::parallel_for_each(
      tf::make_sequence_range(arrangement.n_faces()),
      [&](Index face) {
        const auto range = arrangement.face_range(face);
        for (auto row = range[0]; row < range[1]; ++row)
          face_of_row[std::size_t(row)] = face;
      },
      tf::checked);
  struct local_t {
    tf::buffer<plane_piece_crossing_entry<Index>> entries;
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
        // incidence fills a piece's rows triangle-ascending, so the
        // depth's repeat is the previous row's triangle
        local.entries.clear();
        Index live = 0;
        Index previous = Index(-1);
        for (const auto row : rows) {
          const auto triangle = row / Index(3);
          live += Index(coplanar_of[std::size_t(triangle)] == Index(-1));
          if (triangle == previous)
            continue;
          previous = triangle;
          const auto face = face_of_row[std::size_t(triangle)];
          local.entries.push_back(
              {cells.cell_of[std::size_t(triangle)],
               Index(plane_arrangement_face_descriptor(arrangement, immutable,
                                                       face)
                         .tag),
               Index(plane_arrangement_face_orientation(arrangement,
                                                        immutable, face))});
        }
        if (live > Index(2)) {
          fences.fan[std::size_t(piece)] = char(1);
          fences.crossable[std::size_t(piece)] = char(0);
          return;
        }
        for (const auto &definition :
             arrangement.piece_definitions(immutable, piece)) {
          if ((definition.flags &
               (tf::intersect::graph::plane_edge_fan_flag |
                tf::intersect::graph::plane_edge_non_manifold_flag)) != 0) {
            fences.fan[std::size_t(piece)] = char(1);
            fences.crossable[std::size_t(piece)] = char(0);
            return;
          }
        }
        fences.fan[std::size_t(piece)] = char(0);
        fences.crossable[std::size_t(piece)] =
            char(!plane_piece_depth_varies(local.entries));
      },
      local_t{}, tf::checked);
  return fences;
}

} // namespace tf::arrangement
