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
#include "../../../core/config/verify.hpp"
#include "./charge_tet_lifted_change.hpp"
#include "./flip_tet_edge.hpp"
#include "./flip_tet_facet.hpp"
#include "./admitted_tet_edge.hpp"
#include "./flip_tet_quad.hpp"
#include "./tet_edge_ref.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_facets.hpp"
#include "./tet_flip_constraints.hpp"
#include "./tet_flip_eligible.hpp"
#include "./tet_flip_scratch.hpp"
#include "./tet_lifted_change.hpp"
#include "./tet_orient3d_sites.hpp"
#include "./tet_state.hpp"
#include <array>
#include <cstddef>
#include <initializer_list>

namespace tf::topology::cdt::dt3 {

/// The row among `rows` holding both `x` and `y`.
template <typename Owner, typename Rows>
auto tet_row_holding(const Owner &owner, const Rows &rows,
                     typename Owner::index_type x, typename Owner::index_type y)
    -> typename Owner::index_type {
  for (auto row : rows) {
    const auto c = owner._corners[std::size_t(row)];
    bool hx = false, hy = false;
    for (std::size_t s = 0; s < 4; ++s) {
      hx = hx || c[s] == x;
      hy = hy || c[s] == y;
    }
    if (hx && hy)
      return row;
  }
  if constexpr (tf::core::verification)
    tf::core::verify(false, "some row of the ring holds both corners");
  return Owner::none;
}

/// The handle on `row` whose facet is `x y z`, `x` its org, `z` its apex.
template <typename Owner>
auto tet_face_handle(const Owner &owner, typename Owner::index_type row,
                     typename Owner::index_type x, typename Owner::index_type y,
                     typename Owner::index_type z)
    -> tet_edge_ref<typename Owner::index_type> {
  auto handle = tet_edge_of(owner, row, x, y);
  if (owner._corners[std::size_t(row)][tet_edge_apex(handle)] != z)
    handle = tet_edge_sym(handle);
  if constexpr (tf::core::verification)
    tf::core::verify(owner._corners[std::size_t(row)][tet_edge_apex(handle)] ==
                         z,
                     "the handle names the facet");
  return handle;
}

/// Whether any of the three ring facets a b x of `at`, each the facet of
/// its handle, is protected: then the ring is not flipped. Two protected
/// facets in a row would hold a flat cell between them, which a valid
/// complex never does, so no 2-2 is admitted. A facet is read from BOTH its
/// cells, because the flip these guard removes names the same three facets
/// from its own sides (@ref flip_tet_edge: `a flip removes no protected
/// facet`) and a bit standing on one side alone would pass a guard that read
/// only the other.
template <typename Owner>
auto tet_ring_facets_protected(
    const Owner &owner,
    const std::array<tet_edge_ref<typename Owner::index_type>, 3> &at) -> bool {
  for (const auto &r : at) {
    const std::size_t slot = tet_edge_oppo(r);
    if (tet_facet_is_protected(owner, r.cell, slot))
      return true;
    const auto mate = owner._neighbors[std::size_t(r.cell)][slot];
    if (mate < 0)
      continue;
    std::size_t back = 0;
    while (back < 4 && owner._neighbors[std::size_t(mate)][back] != r.cell)
      ++back;
    if (back < 4 && tet_facet_is_protected(owner, mate, back))
      return true;
  }
  return false;
}

/// Undo a 2-3 that made the ring cell `star` = [a, b, e, d]: the 3-2 of
/// e d about it, charged to `fc.lifted` (@ref charge_tet_lifted_change),
/// returning the two rows, a b c d and a b e c.
template <typename Owner>
auto undo_tet_facet_flip(
    Owner &owner, tet_flip_scratch<typename Owner::index_type> &scratch,
    const tet_edge_ref<typename Owner::index_type> &star,
    const tet_flip_constraints<typename Owner::index_type,
                               typename Owner::coord_type,
                               typename Owner::int_type> &fc)
    -> std::array<typename Owner::index_type, 2> {
  using Index = typename Owner::index_type;
  const auto c = tet_edge_corners(owner, star);
  const auto facet = tet_facet(owner._corners[std::size_t(star.cell)], star.a);
  std::size_t edge = 0;
  while (edge < 3 &&
         !((facet[edge] == c[2] && facet[(edge + 1) % 3] == c[3]) ||
           (facet[edge] == c[3] && facet[(edge + 1) % 3] == c[2])))
    ++edge;
  if constexpr (tf::core::verification)
    tf::core::verify(edge < 3, "the facet holds the edge");
  if (fc.lifted)
    charge_tet_lifted_change(
        owner, *fc.lifted,
        std::initializer_list<Index>{
            star.cell, owner._neighbors[std::size_t(star.cell)][star.a],
            owner._neighbors[std::size_t(star.cell)][star.b]},
        -1);
  const auto rows =
      flip_tet_edge(owner, star.cell, star.a, edge, scratch.round);
  if (fc.lifted)
    charge_tet_lifted_change(owner, *fc.lifted, rows, 1);
  tet_ring_fresh(owner, scratch, rows);
  return rows;
}

/// Seat the two handles the undo of an n-m flip of the link edge `pivot`
/// starts from, at `span`, from the ring cell `star` = [a, b, e, d] the
/// flip made: for c b the handles [d, a, e, b] and [a, d, e, c], for a c
/// [d, b, e, c] and [b, d, e, a].
template <typename Owner>
auto seat_tet_ring_undo(Owner &owner,
                        tet_flip_scratch<typename Owner::index_type> &scratch,
                        std::size_t span, int pivot,
                        const tet_edge_ref<typename Owner::index_type> &star)
    -> void {
  using Index = typename Owner::index_type;
  using entry_t = tet_ring_entry<Index>;
  const auto c = tet_edge_corners(owner, star);
  const auto put = [&scratch](std::size_t at, const tet_edge_ref<Index> &r) {
    auto &e = scratch.arena[at];
    e = entry_t{};
    e.cell = r.cell;
    e.a = r.a;
    e.b = r.b;
  };
  if (pivot == 1) {
    put(span, tet_edge_of(owner, star.cell, c[3], c[0]));
    const Index other = owner._neighbors[std::size_t(star.cell)][star.b];
    put(span + 1, tet_edge_of(owner, other, c[0], c[3]));
  } else {
    put(span + 1, tet_edge_of(owner, star.cell, c[1], c[3]));
    const Index other = owner._neighbors[std::size_t(star.cell)][star.a];
    put(span, tet_edge_of(owner, other, c[3], c[1]));
  }
}

template <typename Owner, typename IsSegment>
auto unflip_tet_ring(Owner &owner,
                     tet_flip_scratch<typename Owner::index_type> &scratch,
                     std::size_t begin, int n, int nn, int ab_pivot,
                     const tet_flip_constraints<typename Owner::index_type,
                                                typename Owner::coord_type,
                                                typename Owner::int_type> &fc,
                     const IsSegment &is_segment) -> void;

/// Flip the edge a b out of the complex through a sequence of elementary flips:
/// the ring's `n` cells in right-hand order about a -> b stand at `begin` in
/// the scratch's arena. While the ring holds more than three cells it is shrunk
/// one apex at a time: a facet a b c whose two neighbouring apexes see each
/// other through it flips 2-3, a flat ring of four flips 4-4 — a hull edge's
/// ring of four too, its hull faces coplanar and its two finite cells valid,
/// read from the hull apex or from the finite facet across whose flat flip the
/// hull apex is the fourth, where the quad's diagonals must cross: the 2-2 on
/// the hull — and, within the link level, a link edge c b or a c that is not
/// locally convex is flipped away by this same operation one level deeper; a
/// ring of three flips 3-2 when its two new cells are valid. A facet that is
/// protected, every other facet of a hull cell, a cell in two rings, a segment,
/// a ring past the star limit and a flip the constraints refuse are each left
/// alone. Without `fc.unflip` a reduction that does not lead to the flip is
/// kept and the reduced size is returned; with it every flip is undone and the
/// search goes on. Under `fc.lifted` every flip is charged and a flip of the
/// edge at level 0 that leaves the lifted volume no lower is undone. Returns 2
/// when the edge is flipped — the arena's first two entries are then the two
/// new cells, the one holding b first, each a handle on the new facet whose
/// corners are the three apexes in the ring's order — and the ring's size
/// otherwise, the entries past the live ones recording the flips made for
/// @ref unflip_tet_ring. `level` is the link level, `ab_pivot` which link edge
/// of the level above this edge is, 0 at the top.
template <typename Owner, typename IsSegment>
auto flip_tet_ring(Owner &owner,
                   tet_flip_scratch<typename Owner::index_type> &scratch,
                   std::size_t begin, int n, int level, int ab_pivot,
                   const tet_flip_constraints<typename Owner::index_type,
                                              typename Owner::coord_type,
                                              typename Owner::int_type> &fc,
                   const IsSegment &is_segment) -> int {
  using Index = typename Owner::index_type;
  using entry_t = tet_ring_entry<Index>;
  using ref_t = tet_edge_ref<Index>;
  auto &arena = scratch.arena;
  const auto entry = [&arena, begin](int i) -> entry_t & {
    return arena[begin + std::size_t(i)];
  };
  const auto live = [&entry](int i) {
    const auto &e = entry(i);
    if constexpr (tf::core::verification)
      tf::core::verify(e.kind == entry_t::live,
                       "a ring entry read as live is live");
    return ref_t{e.cell, e.a, e.b};
  };
  const auto put = [&entry](int i, const ref_t &r) {
    auto &e = entry(i);
    e = entry_t{};
    e.cell = r.cell;
    e.a = r.a;
    e.b = r.b;
  };
  const auto corners = [&owner, &live](int i) {
    return tet_edge_corners(owner, live(i));
  };
  const auto below = [&owner](Index w, Index x, Index y, Index z) {
    return tet_orient3d_below_sites(owner, w, x, y, z);
  };
  const auto face_held = [&owner, &live](int i) {
    const auto r = live(i);
    const std::size_t slot = tet_edge_oppo(r);
    if (!tet_facet_is_protected(owner, r.cell, slot))
      return false;
    return true;
  };
  const auto count = [&owner, &scratch, &entry](int i) -> std::uint8_t & {
    return tet_ring_count(owner, scratch, entry(i).cell);
  };
  const auto before = [&owner, &fc](std::initializer_list<Index> cells) {
    if (fc.lifted)
      charge_tet_lifted_change(owner, *fc.lifted, cells, -1);
  };
  const auto charge = [&owner, &fc](const auto &rows) {
    if (fc.lifted)
      charge_tet_lifted_change(owner, *fc.lifted, rows, 1);
  };
  const auto worse = [&owner, &fc, level]() {
    return fc.lifted && level == 0 &&
           !tet_lifted_change_lowers(owner, *fc.lifted);
  };
  const Index pa = corners(0)[0], pb = corners(0)[1];
  if (n > 3) {
    int reflex = 0;
    for (int i = 0; i < n; ++i) {
      // A STOPPED TASK STANDS WHERE IT IS: the level answers at once, so
      // every level it unwinds through releases what it pushed and the
      // arena never describes a ring the complex no longer has.
      if (scratch.round.stopped)
        return n;
      if (face_held(i))
        continue;
      if (count(i) > 1 || count((i - 1 + n) % n) > 1)
        continue;
      const Index pc = corners(i)[2];
      const Index pd = corners((i + 1) % n)[2];
      const Index pe = corners((i - 1 + n) % n)[2];
      if (pd == Owner::infinite || pe == Owner::infinite)
        continue;
      bool reducible = false;
      int ori = 0;
      if (pc == Owner::infinite) {
        if (n == 4 && below(pa, pb, pd, pe) == 0) {
          const Index pf = corners((i + 2) % n)[2];
          reducible = pf != Owner::infinite && below(pd, pe, pf, pa) < 0 &&
                      below(pe, pd, pf, pb) < 0;
        }
        if (!reducible) {
          ++scratch.hull_barred;
          continue;
        }
      } else {
        ori = below(pb, pc, pd, pe);
        if (ori > 0) {
          ori = below(pc, pa, pd, pe);
          if (ori > 0) {
            ori = below(pa, pb, pd, pe);
            if (ori > 0)
              reducible = true;
            else if (ori == 0 && n == 4)
              reducible = true;
          }
        }
        if (!reducible) {
          ++reflex;
          continue;
        }
      }
      if (!tet_flip_eligible(owner, 1, pa, pb, pc, pd, pe, fc))
        continue;
      if (ori == 0) {
        const Index pf = corners((i + 2) % n)[2];
        bool ok = pf == Owner::infinite
                      ? below(pd, pe, pc, pa) * below(pd, pe, pc, pb) < 0
                      : below(pd, pe, pf, pa) < 0 && below(pe, pd, pf, pb) < 0;
        if (ok) {
          const std::array<tet_edge_ref<Index>, 3> ring{
              live((i - 1 + n) % n), live((i + 1) % n), live((i + 2) % n)};
          if (tet_ring_facets_protected(owner, ring)) {
            ok = false;
          }
        }
        ok = ok && tet_flip_eligible(owner, 2, pe, pd, pf, pb, pa, fc);
        if (!ok) {
          if (!fc.unflip)
            return n;
          continue;
        }
        const std::array<ref_t, 4> ring{live(0), live(1), live(2), live(3)};
        for (const auto &r4 : ring)
          if (!admit_tet_flip_closure(owner, scratch.round, r4.cell))
            return n;
        before({ring[0].cell, ring[1].cell, ring[2].cell, ring[3].cell});
        const auto rows =
            flip_tet_quad(owner, ring, std::size_t(i), scratch.round);
        if (scratch.round.stopped)
          return n;
        charge(rows);
        tet_ring_fresh(owner, scratch, rows);
        if (worse()) {
          std::array<ref_t, 4> about{};
          auto spin = tet_edge_of(owner, rows[0], pe, pd);
          std::size_t at = 4;
          for (std::size_t k = 0; k < 4; ++k) {
            about[k] = spin;
            if (tet_edge_corners(owner, spin)[2] == pc)
              at = k;
            spin = tet_edge_next(owner, spin);
          }
          if constexpr (tf::core::verification)
            tf::core::verify(spin.cell == about[0].cell && at < 4,
                             "the spin closes on the quad after four cells");
          before({about[0].cell, about[1].cell, about[2].cell, about[3].cell});
          const auto back = flip_tet_quad(owner, about, at, scratch.round);
          if (scratch.round.stopped)
            return n;
          charge(back);
          auto walk = tet_edge_of(owner, back[0], pa, pb);
          const Index around = walk.cell;
          while (tet_edge_corners(owner, walk)[2] != pc) {
            walk = tet_edge_next(owner, walk);
            if (walk.cell == around)
              break;
          }
          if constexpr (tf::core::verification)
            tf::core::verify(tet_edge_corners(owner, walk)[2] == pc,
                             "the ring holds the apex");
          for (int m = 0; m < 4; ++m) {
            put((i + m) % 4, walk);
            tet_ring_count(owner, scratch, walk.cell) = 1;
            walk = tet_edge_next(owner, walk);
          }
          if (!fc.unflip)
            return n;
          continue;
        }
        const Index row_b = tet_row_holding(owner, rows, pf, pb);
        const Index row_a = tet_row_holding(owner, rows, pf, pa);
        if (fc.collect) {
          for (auto row : rows)
            if (row != row_a && row != row_b)
              scratch.collected.push_back(tet_edge_of(owner, row, pe, pd));
          if (level == 0) {
            scratch.collected.push_back(tet_edge_of(owner, row_a, pe, pd));
            scratch.collected.push_back(tet_edge_of(owner, row_b, pe, pd));
          } else
            scratch.collected.push_back(
                tet_edge_of(owner, ab_pivot == 1 ? row_a : row_b, pe, pd));
        }
        put(0, tet_face_handle(owner, row_b, pe, pd, pf));
        put(1, tet_face_handle(owner, row_a, pd, pe, pf));
        auto &rec = entry(3);
        rec = entry_t{};
        rec.kind = entry_t::quad_flip;
        rec.at = Index(i);
        rec.apex = pc;
        rec.span = pe;
        rec.n1 = pd;
        return 2;
      }
      const auto r = live(i);
      if (!admit_tet_flip_closure(owner, scratch.round, r.cell) ||
          !admit_tet_flip_closure(owner, scratch.round,
                                  live((i - 1 + n) % n).cell))
        return n;
      before({r.cell, live((i - 1 + n) % n).cell});
      const auto rows =
          flip_tet_facet(owner, r.cell, tet_edge_oppo(r), scratch.round);
      if (scratch.round.stopped)
        return n;
      charge(rows);
      tet_ring_fresh(owner, scratch, rows);
      const Index star_row = tet_row_holding(owner, rows, pa, pb);
      ++tet_ring_count(owner, scratch, star_row);
      put((i - 1 + n) % n, tet_edge_of(owner, star_row, pa, pb));
      for (int j = i; j < n - 1; ++j)
        entry(j) = entry(j + 1);
      {
        auto &rec = entry(n - 1);
        rec = entry_t{};
        rec.kind = entry_t::facet_flip;
        rec.at = Index(i);
        rec.apex = pc;
      }
      if (fc.collect)
        for (auto row : rows)
          if (row != star_row)
            scratch.collected.push_back(tet_edge_of(owner, row, pe, pd));
      int nn = flip_tet_ring(owner, scratch, begin, n - 1, level, ab_pivot, fc,
                             is_segment);
      if (nn == 2)
        return nn;
      if (fc.unflip) {
        const auto star = live((i - 1 + (n - 1)) % (n - 1));
        const auto two = undo_tet_facet_flip(owner, scratch, star, fc);
        for (int j = n - 2; j >= i; --j)
          entry(j + 1) = entry(j);
        const Index row_e = tet_row_holding(owner, two, pe, pc);
        const Index row_d = tet_row_holding(owner, two, pd, pc);
        put((i - 1 + n) % n, tet_edge_of(owner, row_e, pa, pb));
        put(i, tet_edge_of(owner, row_d, pa, pb));
        ++tet_ring_count(owner, scratch, row_e);
        ++tet_ring_count(owner, scratch, row_d);
        ++nn;
        if (fc.collect) {
          scratch.collected.pop_back();
          scratch.collected.pop_back();
        }
      }
      if (!fc.unflip)
        return nn;
    }
    if (reflex > 0 && level < tet_flip_level_limit(scratch)) {
      for (int i = 0; i < n; ++i) {
        if (count(i) > 1 || count((i - 1 + n) % n) > 1)
          continue;
        const Index pc = corners(i)[2];
        if (pc == Owner::infinite)
          continue;
        const Index pd = corners((i + 1) % n)[2];
        const Index pe = corners((i - 1 + n) % n)[2];
        if (pd == Owner::infinite || pe == Owner::infinite)
          continue;
        const auto r = live(i);
        int pivot = 0;
        ref_t flipedge{};
        if (below(pb, pc, pd, pe) <= 0) {
          flipedge = tet_edge_of(owner, r.cell, pc, pb);
          pivot = 1;
        } else if (below(pc, pa, pd, pe) <= 0) {
          flipedge = tet_edge_of(owner, r.cell, pa, pc);
          pivot = 2;
        } else
          continue;
        const auto fe = tet_edge_corners(owner, flipedge);
        if (is_segment(fe[0], fe[1]))
          continue;
        int n1 = 0, j = 0;
        auto spin = flipedge;
        do {
          ++n1;
          j += tet_ring_count(owner, scratch, spin.cell);
          spin = tet_edge_next(owner, spin);
        } while (spin.cell != flipedge.cell);
        if constexpr (tf::core::verification)
          tf::core::verify(n1 >= 3, "a ring has at least three cells");
        if (j > 2)
          continue;
        if constexpr (tf::core::verification)
          tf::core::verify(j == 2,
                           "the flip edge stands in two cells of the ring");
        if (scratch.star_limit > 0 && n1 > scratch.star_limit)
          continue;
        const std::size_t span = arena.size();
        spin = flipedge;
        for (int k = 0; k < n1; ++k) {
          auto &e = arena.push_back(entry_t{});
          e.cell = spin.cell;
          e.a = spin.a;
          e.b = spin.b;
          ++tet_ring_count(owner, scratch, spin.cell);
          spin = tet_edge_next(owner, spin);
        }
        int nn = flip_tet_ring(owner, scratch, span, n1, level + 1, pivot, fc,
                               is_segment);
        if (nn == 2) {
          const Index star_row = arena[span + (pivot == 1 ? 0 : 1)].cell;
          if constexpr (tf::core::verification)
            tf::core::verify(tet_ring_count(owner, scratch, star_row) == 0,
                             "the cell the flip leaves is not yet in the ring");
          ++tet_ring_count(owner, scratch, star_row);
          put((i - 1 + n) % n, tet_edge_of(owner, star_row, pa, pb));
          for (int k = i; k < n - 1; ++k)
            entry(k) = entry(k + 1);
          {
            auto &rec = entry(n - 1);
            rec = entry_t{};
            rec.kind = entry_t::ring_flip;
            rec.at = Index(i);
            rec.pivot = std::uint8_t(pivot);
            rec.span = Index(span);
            rec.n1 = Index(n1);
            rec.apex = pc;
          }
          nn = flip_tet_ring(owner, scratch, begin, n - 1, level, ab_pivot, fc,
                             is_segment);
          if (nn == 2)
            return nn;
          if (fc.unflip) {
            if constexpr (tf::core::verification)
              tf::core::verify(nn == n - 1,
                               "an undone ring flip leaves one cell fewer");
            const auto star = live((i - 1 + (n - 1)) % (n - 1));
            seat_tet_ring_undo(owner, scratch, span, pivot, star);
            unflip_tet_ring(owner, scratch, span, n1, 2, pivot, fc, is_segment);
            for (int k = n - 2; k >= i; --k)
              entry(k + 1) = entry(k);
            Index row_e = Owner::none, row_d = Owner::none;
            for (int k = 0; k < n1; ++k) {
              const Index cell = arena[span + std::size_t(k)].cell;
              const auto c = owner._corners[std::size_t(cell)];
              bool ha = false, hb = false, he = false, hd = false;
              for (std::size_t s = 0; s < 4; ++s) {
                ha = ha || c[s] == pa;
                hb = hb || c[s] == pb;
                he = he || c[s] == pe;
                hd = hd || c[s] == pd;
              }
              if (ha && hb && he)
                row_e = cell;
              if (ha && hb && hd)
                row_d = cell;
            }
            if constexpr (tf::core::verification)
              tf::core::verify(row_e != Owner::none && row_d != Owner::none,
                               "the flip leaves both cells about the star");
            if constexpr (tf::core::verification)
              tf::core::verify(
                  tet_ring_count(owner, scratch, row_e) == 0 &&
                      tet_ring_count(owner, scratch, row_d) == 0,
                  "the cells the flip leaves are not yet in the ring");
            ++tet_ring_count(owner, scratch, row_e);
            ++tet_ring_count(owner, scratch, row_d);
            put((i - 1 + n) % n, tet_edge_of(owner, row_e, pa, pb));
            put(i, tet_edge_of(owner, row_d, pa, pb));
            ++nn;
            arena.erase_till_end(arena.begin() + std::ptrdiff_t(span));
          }
          if (!fc.unflip)
            return nn;
        } else {
          if (fc.unflip) {
            if constexpr (tf::core::verification)
              tf::core::verify(nn == n1,
                               "an undone ring flip restores the ring");
          } else {
            unflip_tet_ring(owner, scratch, span, n1, nn, pivot, fc, is_segment);
          }
          for (int k = 0; k < nn; ++k) {
            auto &c = tet_ring_count(owner, scratch, arena[span + std::size_t(k)].cell);
            if constexpr (tf::core::verification)
              tf::core::verify(c > 0,
                               "a cell leaves the ring only once per entry");
            --c;
          }
          arena.erase_till_end(arena.begin() + std::ptrdiff_t(span));
        }
      }
    }
    return n;
  }
  if (scratch.round.stopped)
    return n;
  const Index pc = corners(0)[2], pd = corners(1)[2], pe = corners(2)[2];
  if (pc == Owner::infinite || pd == Owner::infinite || pe == Owner::infinite) {
    ++scratch.hull_barred;
    return n;
  }
  if (!(below(pd, pc, pe, pa) < 0 && below(pc, pd, pe, pb) < 0))
    return n;
  {
    const std::array<tet_edge_ref<Index>, 3> ring{live(0), live(1), live(2)};
    if (tet_ring_facets_protected(owner, ring)) {
      return n;
    }
  }
  if (!tet_flip_eligible(owner, 2, pc, pd, pe, pb, pa, fc))
    return n;
  const auto r0 = live(0);
  const auto facet =
      tet_facet(owner._corners[std::size_t(r0.cell)], tet_edge_oppo(r0));
  std::size_t edge = 0;
  while (edge < 3 && !((facet[edge] == pa && facet[(edge + 1) % 3] == pb) ||
                       (facet[edge] == pb && facet[(edge + 1) % 3] == pa)))
    ++edge;
  if constexpr (tf::core::verification)
    tf::core::verify(edge < 3, "the facet holds the edge");
  for (int k = 0; k < 3; ++k)
    if (!admit_tet_flip_closure(owner, scratch.round, live(k).cell))
      return n;
  before({live(0).cell, live(1).cell, live(2).cell});
  const auto rows =
      flip_tet_edge(owner, r0.cell, tet_edge_oppo(r0), edge, scratch.round);
  if (scratch.round.stopped)
    return n;
  charge(rows);
  tet_ring_fresh(owner, scratch, rows);
  const Index row_b = tet_row_holding(owner, rows, pb, pc);
  const Index row_a = tet_row_holding(owner, rows, pa, pc);
  if (worse()) {
    const auto cb = owner._corners[std::size_t(row_b)];
    std::size_t slot_b = 0;
    while (slot_b < 4 && cb[slot_b] != pb)
      ++slot_b;
    if constexpr (tf::core::verification)
      tf::core::verify(slot_b < 4, "the row holds the corner");
    before({row_b, row_a});
    const auto back = flip_tet_facet(owner, row_b, slot_b, scratch.round);
    if (scratch.round.stopped)
      return n;
    charge(back);
    tet_ring_fresh(owner, scratch, back);
    put(0, tet_edge_of(owner, tet_row_holding(owner, back, pc, pd), pa, pb));
    put(1, tet_edge_of(owner, tet_row_holding(owner, back, pd, pe), pa, pb));
    put(2, tet_edge_of(owner, tet_row_holding(owner, back, pe, pc), pa, pb));
    for (int k = 0; k < 3; ++k)
      ++count(k);
    return 3;
  }
  if (fc.collect) {
    if (level == 0) {
      scratch.collected.push_back(tet_edge_of(owner, row_b, pc, pd));
      scratch.collected.push_back(tet_edge_of(owner, row_a, pc, pd));
    } else
      scratch.collected.push_back(
          tet_edge_of(owner, ab_pivot == 1 ? row_a : row_b, pc, pd));
  }
  put(0, tet_face_handle(owner, row_b, pc, pd, pe));
  put(1, tet_face_handle(owner, row_a, pd, pc, pe));
  return 2;
}

/// Walk back over the flips @ref flip_tet_ring recorded past the `nn` live
/// entries of the ring at `begin`, freeing the arena spans the link flips took;
/// under `fc.unflip` undoing them too, so the ring stands as before, each entry
/// where it was.
template <typename Owner, typename IsSegment>
auto unflip_tet_ring(Owner &owner,
                     tet_flip_scratch<typename Owner::index_type> &scratch,
                     std::size_t begin, int n, int nn, int ab_pivot,
                     const tet_flip_constraints<typename Owner::index_type,
                                                typename Owner::coord_type,
                                                typename Owner::int_type> &fc,
                     const IsSegment &is_segment) -> void {
  using Index = typename Owner::index_type;
  using entry_t = tet_ring_entry<Index>;
  using ref_t = tet_edge_ref<Index>;
  auto &arena = scratch.arena;
  const auto entry = [&arena, begin](int i) -> entry_t & {
    return arena[begin + std::size_t(i)];
  };
  const auto live = [&entry](int i) {
    const auto &e = entry(i);
    return ref_t{e.cell, e.a, e.b};
  };
  const auto put = [&entry](int i, const ref_t &r) {
    auto &e = entry(i);
    e = entry_t{};
    e.cell = r.cell;
    e.a = r.a;
    e.b = r.b;
  };
  const auto before = [&owner, &fc](std::initializer_list<Index> cells) {
    if (fc.lifted)
      charge_tet_lifted_change(owner, *fc.lifted, cells, -1);
  };
  const auto charge = [&owner, &fc](const auto &rows) {
    if (fc.lifted)
      charge_tet_lifted_change(owner, *fc.lifted, rows, 1);
  };
  if (nn == 2) {
    if (fc.unflip) {
      const auto h0 = live(0);
      const auto hc = tet_edge_corners(owner, h0);
      const Index pb = hc[3];
      const Index pa = tet_edge_corners(owner, live(1))[3];
      if (n >= 4 && entry(3).kind == entry_t::quad_flip) {
        const Index pc = entry(3).apex;
        std::array<ref_t, 4> about{};
        auto spin = tet_edge_of(owner, h0.cell, Index(entry(3).span),
                                Index(entry(3).n1));
        std::size_t at = 4;
        for (std::size_t k = 0; k < 4; ++k) {
          about[k] = spin;
          if (tet_edge_corners(owner, spin)[2] == pc)
            at = k;
          spin = tet_edge_next(owner, spin);
        }
        if constexpr (tf::core::verification)
          tf::core::verify(spin.cell == about[0].cell && at < 4,
                           "the spin closes on the quad after four cells");
        before({about[0].cell, about[1].cell, about[2].cell, about[3].cell});
        const auto rows = flip_tet_quad(owner, about, at, scratch.round);
        charge(rows);
        tet_ring_fresh(owner, scratch, rows);
        const std::size_t t = std::size_t(entry(3).at);
        auto walk = tet_edge_of(owner, rows[0], pa, pb);
        const Index around = walk.cell;
        while (tet_edge_corners(owner, walk)[2] != pc) {
          walk = tet_edge_next(owner, walk);
          if (walk.cell == around)
            break;
        }
        if constexpr (tf::core::verification)
          tf::core::verify(tet_edge_corners(owner, walk)[2] == pc,
                           "the ring holds the apex");
        for (std::size_t m = 0; m < 4; ++m) {
          put(int((t + m) % 4), walk);
          walk = tet_edge_next(owner, walk);
        }
        if (fc.collect)
          for (int k = 0; k < (ab_pivot == 0 ? 4 : 3); ++k)
            scratch.collected.pop_back();
        nn = 4;
      } else {
        const Index pp = hc[0], pq = hc[1], pr = hc[2];
        before({h0.cell, live(1).cell});
        const auto rows =
            flip_tet_facet(owner, h0.cell, tet_edge_oppo(h0), scratch.round);
        charge(rows);
        tet_ring_fresh(owner, scratch, rows);
        put(0, tet_edge_of(owner, tet_row_holding(owner, rows, pp, pq), pa, pb));
        put(1, tet_edge_of(owner, tet_row_holding(owner, rows, pq, pr), pa, pb));
        put(2, tet_edge_of(owner, tet_row_holding(owner, rows, pr, pp), pa, pb));
        if (fc.collect)
          for (int k = 0; k < (ab_pivot == 0 ? 2 : 1); ++k)
            scratch.collected.pop_back();
        nn = 3;
      }
    } else
      nn = 3;
  }
  for (int i = nn; i < n; ++i) {
    const entry_t rec = entry(i);
    if (rec.kind == entry_t::facet_flip) {
      if (!fc.unflip)
        continue;
      const int t = int(rec.at);
      if constexpr (tf::core::verification)
        tf::core::verify(t <= i,
                         "a facet flip undoes at a position the ring holds");
      const auto star = live((t - 1 + i) % i);
      const auto sc = tet_edge_corners(owner, star);
      const auto two = undo_tet_facet_flip(owner, scratch, star, fc);
      for (int j = i - 1; j >= t; --j)
        entry(j + 1) = entry(j);
      put((t - 1 + (i + 1)) % (i + 1),
          tet_edge_of(owner, tet_row_holding(owner, two, sc[2], rec.apex),
                      sc[0], sc[1]));
      put(t, tet_edge_of(owner, tet_row_holding(owner, two, sc[3], rec.apex),
                         sc[0], sc[1]));
      if (fc.collect) {
        scratch.collected.pop_back();
        scratch.collected.pop_back();
      }
    } else if (rec.kind == entry_t::ring_flip) {
      const std::size_t span = std::size_t(rec.span);
      const int n1 = int(rec.n1), pivot = int(rec.pivot), t = int(rec.at);
      if constexpr (tf::core::verification)
        tf::core::verify(t <= i,
                         "a ring flip undoes at a position the ring holds");
      if (fc.unflip) {
        const auto star = live((t - 1 + i) % i);
        const auto sc = tet_edge_corners(owner, star);
        seat_tet_ring_undo(owner, scratch, span, pivot, star);
        unflip_tet_ring(owner, scratch, span, n1, 2, pivot, fc, is_segment);
        for (int j = i - 1; j >= t; --j)
          entry(j + 1) = entry(j);
        Index row_e = Owner::none, row_d = Owner::none;
        for (int k = 0; k < n1; ++k) {
          const Index cell = arena[span + std::size_t(k)].cell;
          const auto c = owner._corners[std::size_t(cell)];
          bool ha = false, hb = false, he = false, hd = false;
          for (std::size_t s = 0; s < 4; ++s) {
            ha = ha || c[s] == sc[0];
            hb = hb || c[s] == sc[1];
            he = he || c[s] == sc[2];
            hd = hd || c[s] == sc[3];
          }
          if (ha && hb && he)
            row_e = cell;
          if (ha && hb && hd)
            row_d = cell;
        }
        if constexpr (tf::core::verification)
          tf::core::verify(row_e != Owner::none && row_d != Owner::none,
                           "the undone flip leaves both cells about the star");
        put((t - 1 + (i + 1)) % (i + 1), tet_edge_of(owner, row_e, sc[0], sc[1]));
        put(t, tet_edge_of(owner, row_d, sc[0], sc[1]));
      } else
        unflip_tet_ring(owner, scratch, span, n1, 2, pivot, fc, is_segment);
      arena.erase_till_end(arena.begin() + std::ptrdiff_t(span));
    }
  }
}

} // namespace tf::topology::cdt::dt3
