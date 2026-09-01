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

#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// CORE. The 2-cell a triangulation face sits in: everything a walk that never
/// crosses a constraint reaches inside this plane. The arrangement's own
/// boundaries are the constraints, so a cell is one of the plane's faces of
/// the arrangement and a filler diagonal never leaves the one it was cut in.
template <typename Index, typename Local, typename Mesh>
auto label_plane_cells(Local &local, const Mesh &mesh) -> void {
  const auto n_triangles = mesh.region_labels().size();
  local.cell_walk.allocate(n_triangles);
  mesh.for_each_face_adjacency([&](Index triangle, Index, Index, Index, Index,
                                   const auto &neighbors,
                                   const auto &constrained) {
    auto &row = local.cell_walk[std::size_t(triangle)];
    for (int s = 0; s < 3; ++s)
      row[std::size_t(s)] = constrained[std::size_t(s)]
                                ? Index(-1)
                                : neighbors[std::size_t(s)];
  });
  local.face_cell.allocate(n_triangles);
  std::fill(local.face_cell.begin(), local.face_cell.end(), Index(-1));
  Index cell = 0;
  for (std::size_t seed = 0; seed < n_triangles; ++seed) {
    if (local.face_cell[seed] != Index(-1))
      continue;
    local.face_cell[seed] = cell;
    local.cell_stack.clear();
    local.cell_stack.push_back(Index(seed));
    while (local.cell_stack.size() != 0) {
      const auto at = local.cell_stack[local.cell_stack.size() - 1];
      local.cell_stack.erase_till_end(local.cell_stack.end() - 1);
      for (const auto peer : local.cell_walk[std::size_t(at)]) {
        if (peer < Index(0) || local.face_cell[std::size_t(peer)] != Index(-1))
          continue;
        local.face_cell[std::size_t(peer)] = cell;
        local.cell_stack.push_back(peer);
      }
    }
    ++cell;
  }
}

/// CORE. The cell a triangle came out of, over the same selection emission
/// walked. The gate is the build's answer, so a plane nobody asked about runs
/// the emit loop it always ran.
template <typename Index, typename Local, typename Mesh>
auto record_single_plane_cells(Local &local, const Mesh &mesh) -> void {
  label_plane_cells<Index>(local, mesh);
  const auto labels = mesh.region_labels();
  for (std::size_t t = 0; t < labels.size(); ++t)
    if (labels[t] != 0)
      local.cell_of.push_back(local.face_cell[t]);
}

/// CORE. A stack's members share one triangulation, so each member carries its
/// own copy of the cells its coverage kept — the same cell, stated once per
/// covering member. The member-major stream IS that selection, in that order.
template <typename Index, typename Local, typename Mesh>
auto record_stack_plane_cells(Local &local, const Mesh &mesh) -> void {
  label_plane_cells<Index>(local, mesh);
  for (const auto triangle : local.member_tri)
    local.cell_of.push_back(local.face_cell[std::size_t(triangle)]);
}

} // namespace tf::arrangement
