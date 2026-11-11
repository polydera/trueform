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
#include "uniform_draw.hpp"
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/tetmesh/identity/account_tet_identities.hpp>
#include <trueform/tetmesh/identity/tet_identity_authority.hpp>
#include <trueform/topology/cdt/dt3/build_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/check_tetrahedralization_validity.hpp>
#include <trueform/topology/cdt/dt3/collapse_tet_site.hpp>
#include <trueform/topology/cdt/dt3/compact_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/resolve_tet_retired_sites.hpp>
#include <trueform/topology/cdt/dt3/tet_state.hpp>
#include <trueform/topology/cdt/dt3/tetrahedralization_owner.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

namespace dt3 = tf::topology::cdt::dt3;
using authority_t = tf::tetmesh::tet_identity_authority<int, std::int64_t>;
using authority_pt = authority_t::pt3_t;
using authority_pair = authority_t::pair_t;
using authority_owner =
    dt3::tetrahedralization_owner<int, std::int64_t, tf::exact::int32>;

auto authority_point(std::int64_t x, std::int64_t y, std::int64_t z)
    -> authority_pt {
  authority_pt p;
  p[0] = x;
  p[1] = y;
  p[2] = z;
  return p;
}

} // namespace

TEST_CASE("tet identity authority: originals, a statement's mints, and the "
          "positions they stand at",
          "[tetmesh][identity]") {
  authority_t authority;
  // originals 0..2, one per position as the entrance states them
  const std::vector<authority_pt> originals{authority_point(0, 0, 0),
                                            authority_point(4, 0, 0),
                                            authority_point(0, 4, 0)};
  const std::vector<int> slots{0, 1, 2};
  authority.name_originals(originals, slots);
  CHECK(authority.n_originals() == 3);
  CHECK(authority.n_created() == 0);
  CHECK(authority.standing_at(authority_point(0, 4, 0)) == 2);
  CHECK(authority.standing_at(authority_point(1, 1, 1)) == -1);
  CHECK(authority.rows().size() == 0);

  // a statement: an original's position, one fresh position twice, another
  // fresh one; the mints follow the first entry of each position
  const std::vector<authority_pt> stated{
      authority_point(9, 9, 9), authority_point(4, 0, 0),
      authority_point(2, 2, 2), authority_point(9, 9, 9)};
  const std::vector<authority_pair> pairs{{5, 7}, {1, 1}, {3, 4}, {2, 8}};
  tf::buffer<int> ids;
  const int first = authority.name_positions(stated, pairs, ids);
  CHECK(first == 3);
  CHECK(authority.n_created() == 2);
  CHECK(ids[1] == 1);
  CHECK(ids[0] == 3);
  CHECK(ids[3] == 3);
  CHECK(ids[2] == 4);
  CHECK(authority.pair_of(3) == authority_pair{2, 8});
  CHECK(authority.pair_of(4) == authority_pair{3, 4});
  CHECK(authority.pair_of(0) == authority_pair{-1, -1});
  CHECK(authority.position_of(4) == authority_point(2, 2, 2));

  // a later statement at a minted position names the standing mint
  tf::buffer<int> again;
  authority.name_positions(std::vector<authority_pt>{authority_point(2, 2, 2)},
                           std::vector<authority_pair>{{0, 0}}, again);
  CHECK(again[0] == 4);
  CHECK(authority.n_created() == 2);
}

TEST_CASE("tet identity authority: the election, a weld, a retirement and "
          "the chase to the standing identity",
          "[tetmesh][identity]") {
  authority_t authority;
  const std::vector<authority_pt> originals{authority_point(0, 0, 0),
                                            authority_point(8, 0, 0),
                                            authority_point(0, 8, 0)};
  authority.name_originals(originals, std::vector<int>{0, 1, 2});
  tf::buffer<int> ids;
  authority.name_positions(std::vector<authority_pt>{authority_point(1, 0, 0),
                                                     authority_point(1, 1, 0)},
                           std::vector<authority_pair>{{0, 1}, {0, 2}}, ids);
  authority.seat(ids[0], 3);
  authority.seat(ids[1], 4);
  CHECK(authority.minted_for(3, {0, 1}));
  CHECK_FALSE(authority.minted_for(3, {0, 2}));
  CHECK_FALSE(authority.minted_for(0, {-1, -1}));
  // a created identity before an original, the smaller of two of a kind
  CHECK(authority.elect(0, ids[0]) == ids[0]);
  CHECK(authority.elect(2, 1) == 1);
  CHECK(authority.elect(ids[1], ids[0]) == ids[0]);
  // the original 1 retires into the mint 4, which welds into the mint 3
  authority.retire(1, ids[1]);
  CHECK(authority.weld(ids[1], ids[0]) == ids[0]);
  CHECK(authority.standing(1) == ids[0]);
  CHECK(authority.target(ids[1]) == ids[0]);
  CHECK(authority.target(ids[0]) == -1);
  CHECK(authority.slot_of(1) == 3);
  CHECK_FALSE(authority.minted_for(4, {0, 2}));
  authority.close();
  REQUIRE(authority.rows().size() == 2);
  CHECK(authority.rows()[0] == std::array<int, 3>{1, 1, ids[0]});
  CHECK(authority.rows()[1] == std::array<int, 3>{1, ids[1], ids[0]});
  // a statement at a retired identity's position names that point, which
  // speaks its survivor
  tf::buffer<int> again;
  authority.name_positions(std::vector<authority_pt>{authority_point(8, 0, 0)},
                           std::vector<authority_pair>{{0, 0}}, again);
  CHECK(again[0] == 1);
  CHECK(authority.standing(again[0]) == ids[0]);
}

TEST_CASE("tet identity authority: a kernel collapse stated as its "
          "retirement, the index map resolved and every identity accounted",
          "[tetmesh][identity]") {
  std::vector<std::int64_t> flat;
  for (int z : {-64, 64})
    for (int y : {-64, 64})
      for (int x : {-64, 64})
        flat.insert(flat.end(), {x, y, z});
  std::mt19937 rng(77u);
  for (std::size_t i = 0; i < 3 * 40; ++i)
    flat.push_back(tf::test::uniform_int(rng, -48, 48));
  authority_owner owner;
  REQUIRE(dt3::build_tetrahedralization(
      owner, tf::make_points<3>(tf::make_range(flat))));
  const std::size_t n = flat.size() / 3;
  std::vector<authority_pt> originals;
  std::vector<int> slots;
  for (std::size_t i = 0; i < n; ++i) {
    originals.push_back(
        authority_point(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]));
    slots.push_back(owner._index_map.f()[i]);
  }
  authority_t authority;
  authority.name_originals(originals, slots);

  // an interior site onto a neighbour the election names its survivor
  bool collapsed = false;
  int site = -1, survivor = -1;
  for (int s = 0; s < int(dt3::tet_site_count(owner)) && !collapsed; ++s) {
    std::vector<int> link;
    bool hull = false;
    for (std::size_t c = 0; c < std::size_t(owner._corners.size()); ++c) {
      if (owner._states[c] == dt3::tet_state::dead)
        continue;
      const auto corners = owner._corners[c];
      bool holds = false;
      for (auto k : corners)
        holds = holds || k == s;
      if (!holds)
        continue;
      hull = hull || corners[3] == authority_owner::infinite;
      for (auto k : corners)
        if (k != s)
          link.push_back(k);
    }
    if (hull)
      continue;
    const int a = authority.identity_at(s);
    for (auto u : link) {
      const int b = authority.identity_at(u);
      if (authority.elect(a, b) != b || !dt3::collapse_tet_site(owner, s, u))
        continue;
      authority.retire(a, b);
      authority.close();
      site = s;
      survivor = u;
      collapsed = true;
      break;
    }
  }
  REQUIRE(collapsed);
  const auto standing = [&authority](int slot) {
    const int id = authority.identity_at(slot);
    return id == -1 ? slot : authority.slot_of(id);
  };
  dt3::resolve_tet_retired_sites(owner, standing);
  for (std::size_t i = 0; i < n; ++i)
    CHECK(owner._index_map.f()[i] != site);
  CHECK(owner._index_map.f()[std::size_t(authority.identity_at(site))] ==
        survivor);
  dt3::compact_tetrahedralization<false, true>(owner);
  const auto retired = [&authority](int slot) {
    const int id = authority.identity_at(slot);
    return id != -1 && authority.target(id) != -1;
  };
  CHECK(dt3::check_tetrahedralization_validity(owner, retired));

  tf::buffer<int> named;
  for (std::size_t c = 0; c < owner._n_finite; ++c)
    for (auto k : owner._corners[c])
      named.push_back(k);
  const int n_sites = int(dt3::tet_site_count(owner));
  const auto accounts =
      tf::tetmesh::account_tet_identities(authority, named, n_sites);
  std::size_t live = 0, gone = 0;
  for (std::size_t id = 0; id < accounts.size(); ++id) {
    if (accounts[id] == tf::tetmesh::tet_identity_account::live)
      ++live;
    if (accounts[id] == tf::tetmesh::tet_identity_account::retired)
      ++gone;
    CHECK(accounts[id] != tf::tetmesh::tet_identity_account::emitted_dead);
  }
  CHECK(gone == 1);
  CHECK(live == n - 1);
  tf::tetmesh::assert_tet_identities_are_accounted(authority, named, n_sites);
  // a published cell still naming the retired site is the break
  named.push_back(site);
  const auto broken =
      tf::tetmesh::account_tet_identities(authority, named, n_sites);
  CHECK(broken[std::size_t(authority.identity_at(site))] ==
        tf::tetmesh::tet_identity_account::emitted_dead);
}

TEST_CASE("tet identity authority: two statements' rows merged equal the "
          "rows of every retirement sorted at once",
          "[tetmesh][identity]") {
  authority_t authority;
  // originals 0..5, each seated at its own site
  const std::vector<authority_pt> originals{
      authority_point(0, 0, 0), authority_point(6, 0, 0),
      authority_point(0, 6, 0), authority_point(0, 0, 6),
      authority_point(6, 6, 0), authority_point(6, 6, 6)};
  authority.name_originals(originals, std::vector<int>{0, 1, 2, 3, 4, 5});
  CHECK(authority.identity_at(4) == 4);

  const auto full_rows = [&authority] {
    std::vector<std::array<int, 3>> rows;
    for (int id = 0; id < authority.n_identities(); ++id)
      if (authority.target(id) != -1)
        rows.push_back({1, id, authority.target(id)});
    return rows;
  };
  const auto stated_rows = [&authority] {
    return std::vector<std::array<int, 3>>(authority.rows().begin(),
                                           authority.rows().end());
  };

  // the first statement: two mints, an original retired into each
  tf::buffer<int> first;
  authority.name_positions(std::vector<authority_pt>{authority_point(3, 0, 0),
                                                     authority_point(0, 3, 0)},
                           std::vector<authority_pair>{{0, 1}, {0, 2}}, first);
  authority.weld(0, first[0]);
  authority.weld(2, first[1]);
  authority.close();
  CHECK(stated_rows() == full_rows());

  // the second: a mint the first statement's survivors retire into, and an
  // original welded onto an earlier survivor, so standing rows retarget
  tf::buffer<int> second;
  authority.name_positions(std::vector<authority_pt>{authority_point(2, 2, 0)},
                           std::vector<authority_pair>{{1, 2}}, second);
  CHECK(authority.weld(second[0], first[0]) == first[0]);
  authority.weld(first[1], first[0]);
  authority.weld(3, first[0]);
  authority.close();
  CHECK(stated_rows() == full_rows());
  CHECK(authority.standing(2) == first[0]);
  for (const auto &row : authority.rows())
    CHECK(authority.target(row[2]) == -1);
}

TEST_CASE("tet identity authority: the exact lattice is the identity",
          "[tetmesh][identity]") {
  authority_t authority;
  const std::vector<authority_pt> originals{authority_point(1, 1, 1),
                                            authority_point(9, 1, 1)};
  authority.name_originals(originals, std::vector<int>{0, 1});
  // a point at an original's position is the original; one a unit away is
  // a point of its own
  tf::buffer<int> ids;
  authority.name_positions(std::vector<authority_pt>{authority_point(1, 1, 1),
                                                     authority_point(2, 1, 1),
                                                     authority_point(2, 1, 1)},
                           std::vector<authority_pair>{{0, 1}, {0, 1}, {0, 2}},
                           ids);
  CHECK(ids[0] == 0);
  CHECK(ids[1] == 2);
  CHECK(ids[2] == 2);
  CHECK(authority.n_created() == 1);
  CHECK(authority.standing_at(authority_point(2, 1, 1)) == 2);
  CHECK(authority.standing_at(authority_point(3, 1, 1)) == -1);
  // a retired point keeps its position, which still names it
  authority.retire(2, 0);
  tf::buffer<int> again;
  authority.name_positions(std::vector<authority_pt>{authority_point(2, 1, 1)},
                           std::vector<authority_pair>{{0, 1}}, again);
  CHECK(again[0] == 2);
  CHECK(authority.standing_at(authority_point(2, 1, 1)) == 0);
  CHECK(authority.position_of(2) == authority_point(2, 1, 1));
}
