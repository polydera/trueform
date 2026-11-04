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
#include "./share_tet_facet.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_facet_key.hpp"
#include "./tet_facets.hpp"
#include "./tet_insertion.hpp"
#include "./tet_orient3d_sites.hpp"
#include "./tet_shared_owners.hpp"
#include "./write_tet.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Insert `site` into a complex that is not Delaunay by splitting the
/// simplex it lies in: `cell`, the finite cell holding it, into four, the
/// facet it lies on into six over the two cells holding it, or the edge it
/// lies on into two cells per cell of the edge's ring. A hull cell among
/// them keeps the hull's winding: its facet is the finite side's read
/// outward, which the orientation sign cannot state for it. No
/// conflict region is read and no wall is met: every facet of the old
/// cells keeps its protection and its owner on the new cell holding it,
/// stated once every link stands, since a reused row's stale links would
/// seat an owner on the wrong facet; a
/// split facet's halves keep both (a shared facet's halves are shared
/// alike), and the edge a ring split divides is the one edge that dies,
/// recorded as killed when the owner records. Returns `placed`, or
/// `capacity` when the rows would pass the index.
template <typename Owner>
auto split_tetrahedralization_site(Owner &owner,
                                   typename Owner::index_type site,
                                   typename Owner::index_type cell)
    -> tet_insertion {
  using Index = typename Owner::index_type;
  using key_t = std::array<Index, 3>;
  struct carried {
    key_t key;
    Index across;
    std::size_t back;
    bool held;
    Index owner;
  };
  struct halved {
    key_t key;
    bool held;
    Index owner;
  };
  const auto orient = [&owner, site](const key_t &f) {
    return tet_orient3d_sites(owner, f[0], f[1], f[2], site);
  };
  const auto read = [&owner](Index c) {
    const auto row = owner._corners[std::size_t(c)];
    return std::array<Index, 4>{row[0], row[1], row[2], row[3]};
  };
  const auto back_slot = [&owner](Index from, Index to) {
    std::size_t s = 0;
    while (owner._neighbors[std::size_t(from)][s] != to)
      ++s;
    return s;
  };
  const auto protection = [&owner](Index cell, std::size_t slot) {
    const bool held = tet_facet_is_protected(owner, cell, slot);
    return std::pair<bool, Index>{
        held, held ? owner._owners[4 * std::size_t(cell) + slot] : Owner::none};
  };
  assert(owner._corners[std::size_t(cell)][3] != Owner::infinite);
  const auto c0 = read(cell);
  std::array<int, 4> sign{};
  std::size_t zeros = 0;
  for (std::size_t s = 0; s < 4; ++s) {
    sign[s] = orient(tet_facet(c0, s));
    assert(sign[s] <= 0);
    zeros += sign[s] == 0;
  }
  assert(zeros < 3);
  tf::small_vector<Index, 32> old;
  tf::small_vector<std::array<Index, 4>, 32> made;
  tf::small_vector<halved, 32> halves;
  key_t edge{Owner::none, Owner::none, Owner::none};
  if (zeros == 0) {
    old.push_back(cell);
    for (std::size_t s = 0; s < 4; ++s) {
      const auto f = tet_facet(c0, s);
      made.push_back({f[0], f[2], f[1], site});
    }
  } else if (zeros == 1) {
    std::size_t z = 0;
    while (sign[z] != 0)
      ++z;
    const Index other = owner._neighbors[std::size_t(cell)][z];
    const auto f = tet_facet(c0, z);
    const auto c1 = read(other);
    std::size_t w = 0;
    while (c1[w] == f[0] || c1[w] == f[1] || c1[w] == f[2])
      ++w;
    old.push_back(cell);
    old.push_back(other);
    for (std::size_t e = 0; e < 3; ++e) {
      const Index a = f[e], b = f[(e + 1) % 3];
      made.push_back({site, a, b, c0[z]});
      made.push_back({site, a, b, c1[w]});
    }
    const auto held = protection(cell, z);
    halves.push_back({tet_facet_key(f), held.first, held.second});
  } else {
    std::size_t z1 = 0;
    while (sign[z1] != 0)
      ++z1;
    std::size_t z2 = z1 + 1;
    while (sign[z2] != 0)
      ++z2;
    Index u = Owner::none, v = Owner::none;
    for (std::size_t s = 0; s < 4; ++s)
      if (s != z1 && s != z2)
        (u == Owner::none ? u : v) = c0[s];
    edge = {std::min(u, v), std::max(u, v), Owner::none};
    Index at = cell, from = Owner::none;
    std::size_t exit = z1;
    for (;;) {
      old.push_back(at);
      const auto c = read(at);
      std::array<Index, 4> under = c, over = c;
      for (std::size_t s = 0; s < 4; ++s) {
        if (c[s] == v)
          under[s] = site;
        if (c[s] == u)
          over[s] = site;
      }
      made.push_back(under);
      made.push_back(over);
      const auto held = protection(at, exit);
      halves.push_back({tet_facet_key(c, exit), held.first, held.second});
      const Index next = owner._neighbors[std::size_t(at)][exit];
      if (next == cell)
        break;
      from = at;
      at = next;
      const auto c_next = read(at);
      const std::size_t entry = back_slot(at, from);
      exit = 4;
      for (std::size_t s = 0; s < 4 && exit == 4; ++s)
        if (s != entry && c_next[s] != u && c_next[s] != v)
          exit = s;
      assert(exit < 4);
    }
  }
  for (auto &m : made) {
    if (m[3] == Owner::infinite)
      continue;
    const int o = tet_orient3d_sites(owner, m[0], m[1], m[2], m[3]);
    assert(o != 0);
    if (o < 0)
      std::swap(m[1], m[2]);
  }
  std::sort(halves.begin(), halves.end(),
            [](const halved &a, const halved &b) { return a.key < b.key; });
  halves.erase(std::unique(halves.begin(), halves.end(),
                           [](const halved &a, const halved &b) {
                             return a.key == b.key;
                           }),
               halves.end());
  const std::size_t first = owner._corners.size();
  const std::size_t fresh = made.size() - old.size();
  if (first + fresh > std::size_t(Owner::max_tets))
    return tet_insertion::capacity;
  tf::small_vector<carried, 64> outer;
  for (auto c : old) {
    const auto corners = read(c);
    for (std::size_t s = 0; s < 4; ++s) {
      const Index across = owner._neighbors[std::size_t(c)][s];
      if (std::find(old.begin(), old.end(), across) != old.end())
        continue;
      const auto held = protection(c, s);
      outer.push_back({tet_facet_key(corners, s), across, back_slot(across, c),
                       held.first, held.second});
    }
  }
  std::sort(outer.begin(), outer.end(),
            [](const carried &a, const carried &b) { return a.key < b.key; });
  owner._corners.reallocate(first + fresh);
  owner._neighbors.reallocate(first + fresh);
  owner._states.reallocate(first + fresh);
  tf::small_vector<Index, 32> rows;
  tf::small_vector<std::pair<Index, std::pair<std::size_t, Index>>, 64> protect;
  for (std::size_t i = 0; i < made.size(); ++i) {
    const Index row = i < old.size() ? old[i] : Index(first + i - old.size());
    rows.push_back(row);
    write_tet(owner, row, made[i][0], made[i][1], made[i][2], made[i][3]);
  }
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto corners = read(rows[i]);
    auto link = owner._neighbors[std::size_t(rows[i])];
    for (std::size_t s = 0; s < 4; ++s) {
      if (link[s] != Owner::none)
        continue;
      const auto key = tet_facet_key(corners, s);
      const auto out = std::lower_bound(
          outer.begin(), outer.end(), key,
          [](const carried &c, const key_t &k) { return c.key < k; });
      if (out != outer.end() && out->key == key) {
        link[s] = out->across;
        owner._neighbors[std::size_t(out->across)][out->back] = rows[i];
        if (out->held)
          protect.push_back({rows[i], {s, out->owner}});
        continue;
      }
      for (std::size_t j = 0; j < rows.size() && link[s] == Owner::none; ++j) {
        if (j == i)
          continue;
        const auto mate = read(rows[j]);
        for (std::size_t t = 0; t < 4; ++t)
          if (tet_facet_key(mate, t) == key) {
            link[s] = rows[j];
            owner._neighbors[std::size_t(rows[j])][t] = rows[i];
            break;
          }
      }
      assert(link[s] != Owner::none);
    }
  }
  for (const auto &h : protect)
    state_tet_facet_protection(owner, h.first, h.second.first, true,
                               h.second.second);
  for (const auto &half : halves) {
    if (!half.held)
      continue;
    tf::small_vector<Index, 4> owners;
    if (half.owner == Owner::shared_owner) {
      const auto stack = tet_shared_owners(owner, half.key);
      for (auto at = stack.first; at != stack.second; ++at)
        owners.push_back(at->second);
      auto &shared = owner._shared;
      shared.erase_till_end(std::copy(
          shared.begin() + (stack.second - shared.begin()), shared.end(),
          shared.begin() + (stack.first - shared.begin())));
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto corners = read(rows[i]);
      for (std::size_t s = 0; s < 4; ++s) {
        const auto key = tet_facet_key(corners, s);
        std::size_t common = 0;
        for (auto k : key)
          common += k == half.key[0] || k == half.key[1] || k == half.key[2];
        if (common != 2 || (key[0] != site && key[1] != site && key[2] != site))
          continue;
        if (tet_facet_is_protected(owner, rows[i], s))
          continue;
        state_tet_facet_protection(owner, rows[i], s, true, half.owner);
        ++owner._n_protected;
        if (half.owner == Owner::shared_owner)
          for (auto o : owners)
            share_tet_facet(owner, key, o);
      }
    }
    --owner._n_protected;
  }
  const bool linked = [&owner, &rows, &read, &back_slot] {
    for (auto row : rows)
      for (std::size_t s = 0; s < 4; ++s) {
        const Index mate = owner._neighbors[std::size_t(row)][s];
        if (mate == Owner::none ||
            tet_facet_key(read(mate), back_slot(mate, row)) !=
                tet_facet_key(read(row), s))
          return false;
      }
    return true;
  }();
  assert(linked);
  (void)linked;
  if (edge[0] != Owner::none && owner._record_killed)
    owner._killed.push_back({edge[0], edge[1]});
  owner._hint = rows[0];
  ++owner._stats.insertions;
  return tet_insertion::placed;
}

} // namespace tf::topology::cdt::dt3
