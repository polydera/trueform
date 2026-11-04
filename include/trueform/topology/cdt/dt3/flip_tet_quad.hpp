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
#include "../../../core/small_vector.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_edge_ref.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_facet_key.hpp"
#include "./tet_facets.hpp"
#include "./tet_orient3d_sites.hpp"
#include "./write_tet.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Flip the edge a b of `ring`, exactly four cells about it in right-hand
/// order with apexes c_0 … c_3, 4-4 at the facet a b c_at: the
/// reference's 2-3 of that facet against the apexes d = c_at+1 and
/// e = c_at-1, which leaves a flat cell a b e d, followed by the 3-2 of
/// a b about e d, d f, f e with f = c_at+2, written as one operation whose
/// flat cell never exists. The four rows become e d b c, e d c a, e d a f,
/// e d f b about the edge e d, linked, then the eight outer facets'
/// protection carried — after every link stands, since the rows are
/// reused and a neighbour's stale link into one would seat the owner on
/// the wrong facet; none of the four facets about a b is protected. A hull
/// edge's two hull cells hold c_at = infinite: the two rows made with it
/// carry it at slot 3 and wind their facet with f on the negative side,
/// the hull law the orientation sign cannot state. The edge a b dies. The
/// caller has read every condition (@ref flip_tet_ring). Returns the four
/// rows in that order, the hint a finite one.
template <typename Owner>
auto flip_tet_quad(Owner &owner,
                   const std::array<tet_edge_ref<typename Owner::index_type>, 4>
                       &ring,
                   std::size_t at)
    -> std::array<typename Owner::index_type, 4> {
  using Index = typename Owner::index_type;
  using key_t = std::array<Index, 3>;
  struct carried {
    key_t key;
    Index across;
    std::size_t back;
    bool protect;
    Index owner;
  };
  const auto first = tet_edge_corners(owner, ring[0]);
  const Index a = first[0], b = first[1];
  std::array<Index, 4> apex{};
  for (std::size_t k = 0; k < 4; ++k)
    apex[k] = tet_edge_corners(owner, ring[k])[2];
  const Index c = apex[at], d = apex[(at + 1) % 4], e = apex[(at + 3) % 4],
              f = apex[(at + 2) % 4];
  std::array<std::array<Index, 4>, 4> made{
      {{e, d, b, c}, {e, d, c, a}, {e, d, a, f}, {e, d, f, b}}};
  for (auto &m : made) {
    if (m[2] == Owner::infinite || m[3] == Owner::infinite) {
      if (m[2] == Owner::infinite)
        std::swap(m[2], m[3]);
      const Index inside = c == Owner::infinite ? f : c;
      const int o = tet_orient3d_sites(owner, m[0], m[1], m[2], inside);
      assert(o != 0);
      if (o > 0)
        std::swap(m[0], m[1]);
      continue;
    }
    const int o = tet_orient3d_sites(owner, m[0], m[1], m[2], m[3]);
    assert(o != 0);
    if (o < 0)
      std::swap(m[2], m[3]);
  }
  const auto back_slot = [&owner](Index from, Index to) {
    std::size_t s = 0;
    while (owner._neighbors[std::size_t(from)][s] != to)
      ++s;
    return s;
  };
  const auto has = [](const key_t &facet, Index corner) {
    return facet[0] == corner || facet[1] == corner || facet[2] == corner;
  };
  tf::small_vector<carried, 8> outer;
  for (std::size_t k = 0; k < 4; ++k) {
    const Index cell = ring[k].cell;
    const auto corners = owner._corners[std::size_t(cell)];
    for (std::size_t s = 0; s < 4; ++s) {
      const auto facet = tet_facet(corners, s);
      if (has(facet, a) && has(facet, b)) {
        assert(!tet_facet_is_protected(owner, cell, s));
        continue;
      }
      const Index across = owner._neighbors[std::size_t(cell)][s];
      const bool protect = tet_facet_is_protected(owner, cell, s);
      outer.push_back({tet_facet_key(facet), across, back_slot(across, cell),
                       protect,
                       protect ? owner._owners[4 * std::size_t(cell) + s]
                               : Owner::none});
    }
  }
  assert(outer.size() == 8);
  std::sort(outer.begin(), outer.end(),
            [](const carried &x, const carried &y) { return x.key < y.key; });
  std::array<Index, 4> rows{ring[0].cell, ring[1].cell, ring[2].cell,
                            ring[3].cell};
  for (std::size_t k = 0; k < 4; ++k)
    write_tet(owner, rows[k], made[k][0], made[k][1], made[k][2], made[k][3]);
  struct held {
    Index row;
    std::size_t slot;
    Index owner;
  };
  tf::small_vector<held, 8> protect;
  for (std::size_t k = 0; k < 4; ++k) {
    const auto corners = owner._corners[std::size_t(rows[k])];
    auto link = owner._neighbors[std::size_t(rows[k])];
    for (std::size_t s = 0; s < 4; ++s) {
      if (link[s] != Owner::none)
        continue;
      const auto key = tet_facet_key(corners, s);
      const auto out = std::lower_bound(
          outer.begin(), outer.end(), key,
          [](const carried &x, const key_t &y) { return x.key < y; });
      if (out != outer.end() && out->key == key) {
        link[s] = out->across;
        owner._neighbors[std::size_t(out->across)][out->back] = rows[k];
        if (out->protect)
          protect.push_back({rows[k], s, out->owner});
        continue;
      }
      for (std::size_t j = 0; j < 4 && link[s] == Owner::none; ++j) {
        if (j == k)
          continue;
        const auto mate = owner._corners[std::size_t(rows[j])];
        for (std::size_t t = 0; t < 4; ++t)
          if (tet_facet_key(mate, t) == key) {
            link[s] = rows[j];
            owner._neighbors[std::size_t(rows[j])][t] = rows[k];
            break;
          }
      }
      assert(link[s] != Owner::none);
    }
  }
  for (const auto &h : protect)
    state_tet_facet_protection(owner, h.row, h.slot, true, h.owner);
  if (owner._record_killed)
    owner._killed.push_back({std::min(a, b), std::max(a, b)});
  owner._hint = rows[c == Owner::infinite ? 2 : 0];
  owner._delaunay = false;
  ++owner._stats.quad_flips;
  return rows;
}

} // namespace tf::topology::cdt::dt3
