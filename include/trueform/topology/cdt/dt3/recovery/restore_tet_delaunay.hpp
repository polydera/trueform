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
#include "../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../../core/views/drop.hpp"
#include "../find_tet_edge.hpp"
#include "../flip_tet_edge.hpp"
#include "../flip_tet_facet.hpp"
#include "../flip_tet_quad.hpp"
#include "../flip_tet_ring.hpp"
#include "../remove_tet_edge_by_flips.hpp"
#include "../tet_edge_ref.hpp"
#include "../tet_edge_slot.hpp"
#include "../tet_facet_is_protected.hpp"
#include "../tet_facets.hpp"
#include "../tet_flip_constraints.hpp"
#include "../tet_flip_scratch.hpp"
#include "../tet_insphere_conflict_sites.hpp"
#include "../tet_lifted_change.hpp"
#include "../tet_orient3d_sites.hpp"
#include "../tet_state.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_stuck_facet.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Restore the complex `owner` to Delaunay wherever flips reach, after a
/// phase's flips and splits left it otherwise. Every facet of a row the phase
/// wrote (the owner's recorded rows) is read in a Lawson pass: one not locally
/// Delaunay flips 2-3 where its two apexes see each other through it, 3-2
/// where its reflex edge has three cells about it and the two new cells are
/// valid, 4-4 where that edge is flat against the apexes and has four; a
/// protected facet, a hull facet, a segment and a ring that admits no flip are
/// left standing, the last recorded; the facets a flip makes are read in turn,
/// and the pass repeats over the facets recorded while it still flips. Then
/// each recorded facet's reflex edge is flipped away by @ref flip_tet_ring at
/// link level one over rings of at most ten cells, every flip undone unless
/// the edge goes and the lifted volume falls (@ref tet_lifted_change); the
/// cells a removal made are read by the Lawson pass again, and what that pass
/// records joins the queue. An edge whose ring stands as at its last failed
/// removal is not tried again. Serial, in the rows' writing order, so the
/// complex is the same at any worker count. Returns the recorded edges no flip
/// removed.
template <typename Owner, typename Index, typename Coord>
auto restore_tet_delaunay(Owner &owner,
                          tet_segment_recovery<Index, Coord> &recovery)
    -> std::size_t {
  using Int = typename Owner::int_type;
  using ref_t = tet_edge_ref<Index>;
  if (owner._delaunay)
    return 0;
  auto &tickets = recovery.tickets;
  auto &stuck = recovery.stuck;
  auto &queue = recovery.stuck_next;
  auto &scratch = recovery.flips;
  tickets.clear();
  stuck.clear();
  queue.clear();
  const auto push = [&tickets](Index row) {
    for (std::size_t slot = 0; slot < 4; ++slot)
      tickets.push_back(Index(4 * std::size_t(row) + slot));
  };
  for (auto row : owner._written)
    if (owner._states[std::size_t(row)] == tet_state::live)
      push(row);
  const auto read = [&owner](Index c) {
    const auto row = owner._corners[std::size_t(c)];
    return std::array<Index, 4>{row[0], row[1], row[2], row[3]};
  };
  const auto below = [&owner](Index w, Index x, Index y, Index z) {
    return tet_orient3d_below_sites(owner, w, x, y, z);
  };
  const auto conflicts = [&owner](const std::array<Index, 4> &corners,
                                  Index apex) {
    return tet_insphere_conflict_sites(owner, corners[0], corners[1],
                                       corners[2], corners[3], apex);
  };
  const auto is_segment = [&recovery](Index p, Index q) {
    const Index low = std::min(p, q), high = std::max(p, q);
    const auto &heads = recovery.row_heads;
    if (std::size_t(low) >= heads.size())
      return false;
    for (Index t = heads[std::size_t(low)]; t != Index(-1);
         t = recovery.row_tickets[std::size_t(t)].second) {
      const auto &row = recovery.subsegments[std::size_t(
          recovery.row_tickets[std::size_t(t)].first)];
      if ((row.a == low && row.b == high) || (row.a == high && row.b == low))
        return true;
    }
    return false;
  };
  const auto standing = [&owner, &read](const tet_stuck_facet<Index> &s,
                                        std::size_t &slot) {
    if (owner._states[std::size_t(s.cell)] != tet_state::live)
      return false;
    const auto c = read(s.cell);
    slot = 4;
    for (std::size_t k = 0; k < 4; ++k)
      if (c[k] != s.corners[0] && c[k] != s.corners[1] && c[k] != s.corners[2])
        slot = slot == 4 ? k : 5;
    return slot < 4;
  };
  const auto lawson = [&]() {
    for (;;) {
      std::size_t flips = 0;
      while (tickets.size()) {
        const Index ticket = tickets.back();
        tickets.pop_back();
        const Index cell = Index(std::size_t(ticket) / 4);
        const std::size_t slot = std::size_t(ticket) % 4;
        if (owner._states[std::size_t(cell)] != tet_state::live)
          continue;
        const Index other = owner._neighbors[std::size_t(cell)][slot];
        if (owner._states[std::size_t(other)] != tet_state::live ||
            tet_facet_is_protected(owner, cell, slot))
          continue;
        const auto c0 = read(cell);
        const auto f = tet_facet(c0, slot);
        const Index pd = c0[slot];
        const auto c1 = read(other);
        std::size_t eslot = 0;
        while (c1[eslot] == f[0] || c1[eslot] == f[1] || c1[eslot] == f[2])
          ++eslot;
        const Index pe = c1[eslot];
        if (!conflicts(c0, pe))
          continue;
        auto h = tet_face_handle(owner, cell, f[0], f[1], f[2]);
        int ori = 1;
        for (int k = 0; k < 3; ++k) {
          const auto hc = tet_edge_corners(owner, h);
          ori = below(hc[0], hc[1], pd, pe);
          if (ori <= 0)
            break;
          h = tet_edge_turn(h);
        }
        if (ori > 0) {
          for (auto row : flip_tet_facet(owner, cell, slot))
            push(row);
          ++flips;
          continue;
        }
        const auto hc = tet_edge_corners(owner, h);
        if (is_segment(hc[0], hc[1]))
          continue;
        int n = 0;
        auto spin = h;
        do {
          ++n;
          spin = tet_edge_next(owner, spin);
        } while (spin.cell != h.cell);
        const auto r1 = tet_edge_next(owner, h);
        const auto r2 = tet_edge_next(owner, r1);
        if (n == 3) {
          const Index x = hc[2], y = tet_edge_corners(owner, r1)[2],
                      z = tet_edge_corners(owner, r2)[2];
          if (x != Owner::infinite && y != Owner::infinite &&
              z != Owner::infinite && below(y, x, z, hc[0]) < 0 &&
              below(x, y, z, hc[1]) < 0 &&
              !tet_ring_facets_protected(owner, {h, r1, r2})) {
            const auto facet = tet_facet(read(h.cell), tet_edge_oppo(h));
            std::size_t e = 0;
            while (!((facet[e] == hc[0] && facet[(e + 1) % 3] == hc[1]) ||
                     (facet[e] == hc[1] && facet[(e + 1) % 3] == hc[0])))
              ++e;
            for (auto row : flip_tet_edge(owner, h.cell, tet_edge_oppo(h), e))
              push(row);
            ++flips;
            continue;
          }
        } else if (n == 4 && ori == 0) {
          const auto r3 = tet_edge_next(owner, r2);
          const Index pc = hc[2];
          const Index x1 = tet_edge_corners(owner, r2)[2];
          const Index x2 = tet_edge_corners(owner, r3)[2];
          if constexpr (tf::core::verification)
            tf::core::verify(tet_edge_corners(owner, r1)[2] == pd && x2 == pe,
                             "a flat ring of four stands about the facet");
          bool ok = pc != Owner::infinite && x1 != Owner::infinite &&
                    x2 != Owner::infinite;
          ok = ok && below(hc[1], pc, pd, pe) > 0 && below(pc, hc[0], pd, pe) > 0;
          ok = ok && below(pd, pe, x1, hc[0]) < 0 && below(pe, pd, x1, hc[1]) < 0;
          ok = ok && !tet_ring_facets_protected(owner, {r1, r2, r3});
          if (ok) {
            for (auto row : flip_tet_quad(owner, {h, r1, r2, r3}, 0))
              push(row);
            ++flips;
            continue;
          }
        }
        stuck.push_back({h.cell, {hc[0], hc[1], hc[2]}});
      }
      if (!stuck.size() || !flips)
        break;
      for (const auto &s : stuck) {
        std::size_t slot = 4;
        if (standing(s, slot))
          tickets.push_back(Index(4 * std::size_t(s.cell) + slot));
      }
      stuck.clear();
    }
  };
  lawson();
  if (!stuck.size())
    return 0;
  tet_flip_constraints<Index, Coord, Int> fc;
  fc.unflip = true;
  fc.collect = true;
  tet_lifted_change<Int, Coord> delta{};
  fc.lifted = &delta;
  scratch.arena.clear();
  scratch.collected.clear();
  scratch.star_limit = 10;
  scratch.level_limit = -1;
  scratch.auto_level = 1;
  std::swap(queue, stuck);
  stuck.clear();
  auto &tried = recovery.tried;
  const auto grow = [&tried](std::size_t rows) {
    const std::size_t known = tried.size();
    if (rows <= known)
      return;
    tried.reallocate(rows);
    tf::parallel_fill(tf::drop(tried, known), std::uint8_t(0));
  };
  tried.clear();
  tried.reserve(2 * owner._corners.size());
  grow(owner._corners.size());
  std::size_t swept = owner._written.size();
  const auto settle = [&owner, &tried, &grow, &swept]() {
    grow(owner._corners.size());
    for (; swept < owner._written.size(); ++swept)
      tried[std::size_t(owner._written[swept])] = 0;
  };
  const auto bit = [](const ref_t &e) {
    return std::uint8_t(1u << tet_edge_slot(e));
  };
  const auto repeats = [&owner, &tried, &bit](const ref_t &edge) {
    auto spin = edge;
    do {
      if (!(tried[std::size_t(spin.cell)] & bit(spin)))
        return false;
      spin = tet_edge_next(owner, spin);
    } while (spin.cell != edge.cell);
    return true;
  };
  const auto mark = [&owner, &tried, &bit](const ref_t &edge) {
    auto spin = edge;
    do {
      tried[std::size_t(spin.cell)] |= bit(spin);
      spin = tet_edge_next(owner, spin);
    } while (spin.cell != edge.cell);
  };
  std::size_t left = 0;
  for (std::size_t i = 0; i < queue.size(); ++i) {
    const auto rec = queue[i];
    ref_t edge{};
    if (!find_tet_edge(owner, rec.corners[0], rec.corners[1], edge))
      continue;
    settle();
    if (repeats(edge)) {
      ++left;
      continue;
    }
    delta = {};
    scratch.collected.clear();
    if (remove_tet_edge_by_flips(owner, scratch, edge, fc, is_segment) == 2) {
      for (const auto &made : scratch.collected)
        if (owner._states[std::size_t(made.cell)] == tet_state::live)
          push(made.cell);
      scratch.collected.clear();
      lawson();
      for (const auto &s : stuck)
        queue.push_back(s);
      stuck.clear();
    } else {
      if constexpr (tf::core::verification) {
        const auto c = tet_edge_corners(owner, edge);
        tf::core::verify(owner._states[std::size_t(edge.cell)] ==
                                 tet_state::live &&
                             c[0] == rec.corners[0] && c[1] == rec.corners[1],
                         "a failed removal hands back its record's edge");
      }
      settle();
      mark(edge);
      ++left;
    }
  }
  scratch.star_limit = -1;
  return left;
}

} // namespace tf::topology::cdt::dt3
