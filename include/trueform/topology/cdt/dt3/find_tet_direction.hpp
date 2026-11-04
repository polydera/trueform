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
#include "./tet_edge_ref.hpp"
#include "./tet_orient3d_sites.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// What the path from a cell's org toward a site meets first in that cell.
enum class tet_direction : std::uint8_t {
  across_vertex,
  across_edge,
  across_face,
  hull,
  lost
};

/// Find the cell on the path from the org a of `edge` to `end`: `edge` keeps
/// its org and ends as [a, b, c, d] with `across_vertex` when a b is collinear
/// with the path, b the end or a site on it; `across_edge` when the edge b c
/// meets it; `across_face` when the facet b c d does; `hull` when the walk
/// left the finite complex, which a convex complex never does. Where several
/// cells may be entered the next is drawn as the reference draws it, from its
/// own congruential generator, seeded here by the walk's start and end so the
/// walk is the same at any worker count; a fixed choice can circle in a
/// complex that is not Delaunay, the draw does not, and a walk that still
/// outlasts twice the cell count is `lost`.
template <typename Owner>
auto find_tet_direction(const Owner &owner,
                        tet_edge_ref<typename Owner::index_type> &edge,
                        typename Owner::index_type end) -> tet_direction {
  using Index = typename Owner::index_type;
  const auto below = [&owner](Index w, Index x, Index y, Index z) {
    return tet_orient3d_below_sites(owner, w, x, y, z);
  };
  const Index pa = owner._corners[std::size_t(edge.cell)][edge.a];
  if (owner._corners[std::size_t(edge.cell)][3] == Owner::infinite) {
    const Index mate = owner._neighbors[std::size_t(edge.cell)][3];
    const auto corners = owner._corners[std::size_t(mate)];
    std::size_t other = corners[0] == pa ? 1 : 0;
    edge = tet_edge_of(owner, mate, pa, corners[other]);
  }
  auto c = tet_edge_corners(owner, edge);
  if (c[1] == end)
    return tet_direction::across_vertex;
  if (c[2] == end) {
    edge = tet_edge_turn_back_sym(edge);
    return tet_direction::across_vertex;
  }
  std::uint32_t seed =
      (std::uint32_t(edge.cell) * 2654435761u) ^ std::uint32_t(end);
  const auto draw = [&seed](std::uint32_t choices) {
    seed = (seed * 1366u + 150889u) % 714025u;
    return seed / (714025u / choices + 1);
  };
  const std::size_t bound = 2 * owner._corners.size() + 16;
  for (std::size_t steps = 0;; ++steps) {
    if (steps > bound)
      return tet_direction::lost;
    c = tet_edge_corners(owner, edge);
    const Index pb = c[1], pc = c[2], pd = c[3];
    if (pd == end) {
      edge = tet_edge_turn(tet_edge_sym(edge));
      return tet_direction::across_vertex;
    }
    if (pd == Owner::infinite)
      return tet_direction::hull;
    const int hori = below(pa, pb, pc, end);
    const int rori = below(pb, pa, pd, end);
    const int lori = below(pa, pc, pd, end);
    enum move { horizon, right, left };
    move next = horizon;
    bool moving = true;
    if (hori > 0) {
      if (rori > 0) {
        if (lori > 0) {
          const auto s = draw(3);
          next = s == 0 ? horizon : s == 1 ? right : left;
        } else
          next = draw(2) ? horizon : right;
      } else
        next = lori > 0 ? (draw(2) ? horizon : left) : horizon;
    } else if (rori > 0)
      next = lori > 0 ? (draw(2) ? right : left) : right;
    else if (lori > 0)
      next = left;
    else
      moving = false;
    if (moving) {
      if (next == right)
        edge = tet_edge_next(owner, edge);
      else if (next == left)
        edge = tet_edge_turn(tet_edge_next(owner, tet_edge_turn_back(edge)));
      else
        edge = tet_edge_turn(tet_edge_across(owner, edge));
    } else {
      if (hori == 0) {
        if (rori == 0)
          return tet_direction::across_vertex;
        if (lori == 0) {
          edge = tet_edge_turn_back_sym(edge);
          return tet_direction::across_vertex;
        }
        return tet_direction::across_edge;
      }
      if (rori == 0) {
        edge = tet_edge_turn(tet_edge_sym(edge));
        return lori == 0 ? tet_direction::across_vertex
                         : tet_direction::across_edge;
      }
      if (lori == 0) {
        edge = tet_edge_turn_back_sym(edge);
        return tet_direction::across_edge;
      }
      return tet_direction::across_face;
    }
    assert(owner._corners[std::size_t(edge.cell)][edge.a] == pa);
  }
}

} // namespace tf::topology::cdt::dt3
