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

#include "../../arrangement/planes/make_plane_arrangement_cells.hpp"
#include "../../arrangement/planes/make_plane_piece_incidence.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/connected_component_labels.hpp"
#include "../../topology/label_connected_components.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// The arrangement's components, under the fence.
///
/// The flood walks CELLS: a crossable piece joins every cell it bounds, a
/// fenced one joins nothing, and each triangle inherits the label of the cell
/// it came out of — a coplanar duplicate with its survivor's, since they are
/// the same cell.
///
/// A label is a NAME, not a fact: only equality between labels means
/// anything, and the finder's walkers claim names in arrival order, so the
/// values themselves carry no order and are not stable across runs.
template <typename Index, typename LabelType = Index>
auto label_plane_arrangement_components(
    const tf::arrangement::plane_piece_incidence<Index> &incidence,
    const tf::buffer<char> &crossable,
    const tf::arrangement::plane_arrangement_cells<Index> &cells)
    -> tf::connected_component_labels<LabelType> {
  // the finder names the cells its walk reached and leaves the rest as it
  // found them, so the buffer is total before it runs
  tf::buffer<LabelType> cell_labels;
  cell_labels.allocate(std::size_t(cells.n_cells));
  tf::parallel_fill(cell_labels, LabelType(-1));
  const auto applier = [&](Index cell, const auto &push) {
    Index previous = Index(-1);
    for (const auto piece : incidence.pieces_of_cell[std::size_t(cell)]) {
      if (piece == previous)
        continue;
      previous = piece;
      if (crossable[std::size_t(piece)] == char(0))
        continue;
      for (const auto row : incidence.rows_of_piece[std::size_t(piece)])
        push(cells.cell_of[std::size_t(row / Index(3))]);
    }
  };
  tf::connected_component_labels<LabelType> out;
  out.n_components =
      tf::label_connected_components<Index>(cell_labels, applier);
  out.labels.allocate(cells.cell_of.size());
  tf::parallel_for_each(
      tf::make_sequence_range(cells.cell_of.size()),
      [&](std::size_t triangle) {
        out.labels[triangle] =
            cell_labels[std::size_t(cells.cell_of[triangle])];
      },
      tf::checked);
  return out;
}

} // namespace tf::csg::graph
