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
#include <trueform/core/none.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/views/offset_block_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/topology/cdt/dt3/build_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/check_tetrahedralization_validity.hpp>
#include <trueform/topology/cdt/dt3/collapse_tet_site.hpp>
#include <trueform/topology/cdt/dt3/compact_tet_dead_rows.hpp>
#include <trueform/topology/cdt/dt3/compact_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/recovery/check_tet_face_fixpoint.hpp>
#include <trueform/topology/cdt/dt3/recovery/collapse_tet_site_rows.hpp>
#include <trueform/topology/cdt/dt3/recovery/recover_tet_face_phase.hpp>
#include <trueform/topology/cdt/dt3/recovery/recover_tet_segment_phase.hpp>
#include <trueform/topology/cdt/dt3/recovery/recover_tet_segments.hpp>
#include <trueform/topology/cdt/dt3/recovery/retire_tet_segments.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_face_recovery.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_face_status.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_piece_row.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_phase.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_recovery.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_status.hpp>
#include <trueform/topology/cdt/dt3/resolve_tet_retired_sites.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_face_rim_sites.hpp>
#include <trueform/topology/cdt/dt3/retire_tet_faces.hpp>
#include <trueform/topology/cdt/dt3/state_tet_facet_protection.hpp>
#include <trueform/topology/cdt/dt3/tet_facet_is_protected.hpp>
#include <trueform/topology/cdt/dt3/tet_facets.hpp>
#include <trueform/topology/cdt/dt3/tet_owner_complex.hpp>
#include <trueform/topology/cdt/dt3/tet_state.hpp>
#include <trueform/topology/cdt/dt3/tetrahedralization_owner.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

namespace dt3 = tf::topology::cdt::dt3;
using collapse_owner =
    dt3::tetrahedralization_owner<int, std::int64_t, tf::exact::int32>;
using collapse_dt = dt3::tet_owner_complex<false, collapse_owner, tf::none_t>;
using collapse_recovery = dt3::tet_segment_recovery<int, std::int64_t>;

/// The cube of half side 64's corners and `n` sites drawn inside it, each
/// given twice, so the weld names its slot by two inputs.
auto collapse_sites(std::uint32_t seed, std::size_t n)
    -> std::vector<std::int64_t> {
  std::vector<std::int64_t> flat;
  for (int z : {-64, 64})
    for (int y : {-64, 64})
      for (int x : {-64, 64})
        flat.insert(flat.end(), {x, y, z});
  std::mt19937 rng(seed);
  for (std::size_t i = 0; i < 3 * n; ++i)
    flat.push_back(tf::test::uniform_int(rng, -48, 48));
  const std::size_t drawn = flat.size();
  for (std::size_t i = 24; i < drawn; ++i)
    flat.push_back(flat[i]);
  return flat;
}

/// Six times the volume the live finite cells hold, exact on the lattice.
auto collapse_volume6(const collapse_owner &owner) -> std::int64_t {
  std::int64_t total = 0;
  for (std::size_t c = 0; c < std::size_t(owner._corners.size()); ++c) {
    if (owner._states[c] == dt3::tet_state::dead ||
        owner._corners[c][3] == collapse_owner::infinite)
      continue;
    const auto p = [&owner, c](std::size_t k) {
      return owner._floors[std::size_t(owner._corners[c][k])].pt;
    };
    const auto a = p(0), b = p(1), d = p(2), e = p(3);
    const std::int64_t u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const std::int64_t v[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
    const std::int64_t w[3] = {e[0] - a[0], e[1] - a[1], e[2] - a[2]};
    total += u[0] * (v[1] * w[2] - v[2] * w[1]) -
             u[1] * (v[0] * w[2] - v[2] * w[0]) +
             u[2] * (v[0] * w[1] - v[1] * w[0]);
  }
  return total;
}

/// The rows, links, states and protections as they stand.
struct collapse_snapshot {
  std::vector<int> corners, neighbors, owners;
  std::vector<std::uint8_t> states, slots;
  std::size_t held = 0;
  auto operator==(const collapse_snapshot &o) const -> bool {
    return corners == o.corners && neighbors == o.neighbors &&
           owners == o.owners && states == o.states && slots == o.slots &&
           held == o.held;
  }
};

auto collapse_snapshot_of(const collapse_owner &owner) -> collapse_snapshot {
  collapse_snapshot s;
  for (auto v : owner._corners.data_buffer())
    s.corners.push_back(v);
  for (auto v : owner._neighbors.data_buffer())
    s.neighbors.push_back(v);
  for (auto v : owner._owners)
    s.owners.push_back(v);
  for (auto v : owner._states)
    s.states.push_back(std::uint8_t(v));
  for (auto v : owner._protected_slots)
    s.slots.push_back(v);
  s.held = owner._n_held;
  return s;
}

/// A cell holding `site` and whether `site` stands on the hull.
auto collapse_star_of(const collapse_owner &owner, int site, bool &hull)
    -> std::vector<int> {
  std::vector<int> star;
  hull = false;
  for (std::size_t c = 0; c < std::size_t(owner._corners.size()); ++c) {
    if (owner._states[c] == dt3::tet_state::dead)
      continue;
    const auto corners = owner._corners[c];
    if (std::find(corners.begin(), corners.end(), site) == corners.end())
      continue;
    hull = hull || corners[3] == collapse_owner::infinite;
    star.push_back(int(c));
  }
  return star;
}

/// The protection bits the live rows hold.
auto collapse_held_bits(const collapse_owner &owner) -> std::size_t {
  std::size_t bits = 0;
  for (std::size_t c = 0; c < std::size_t(owner._corners.size()); ++c)
    if (owner._states[c] != dt3::tet_state::dead &&
        c < owner._protected_slots.size())
      for (std::size_t k = 0; k < 4; ++k)
        bits += (owner._protected_slots[c] >> k) & 1u;
  return bits;
}

auto collapse_built(std::uint32_t seed, std::size_t n, collapse_owner &owner)
    -> void {
  const auto flat = collapse_sites(seed, n);
  REQUIRE(dt3::build_tetrahedralization(
      owner, tf::make_points<3>(tf::make_range(flat))));
}

} // namespace

TEST_CASE("collapse_tet_site: an interior site retires onto a neighbour",
          "[topology][delaunay3][collapse]") {
  collapse_owner owner;
  collapse_built(17u, 40, owner);
  const std::int64_t volume = collapse_volume6(owner);
  std::size_t collapsed = 0, refused = 0;
  for (int site = 0; site < int(dt3::tet_site_count(owner)) && !collapsed;
       ++site) {
    bool hull = false;
    const auto star = collapse_star_of(owner, site, hull);
    if (hull)
      continue;
    std::vector<int> neighbours;
    for (auto c : star)
      for (auto corner : owner._corners[std::size_t(c)])
        if (corner != site)
          neighbours.push_back(corner);
    std::sort(neighbours.begin(), neighbours.end());
    neighbours.erase(std::unique(neighbours.begin(), neighbours.end()),
                     neighbours.end());
    for (auto survivor : neighbours) {
      const auto before = collapse_snapshot_of(owner);
      if (!dt3::collapse_tet_site(owner, site, survivor)) {
        // a refusal changes nothing
        CHECK(collapse_snapshot_of(owner) == before);
        ++refused;
        continue;
      }
      ++collapsed;
      bool gone = false;
      CHECK(collapse_star_of(owner, site, gone).empty());
      std::vector<std::size_t> named;
      for (std::size_t i = 0; i < owner._index_map.f().size(); ++i)
        if (owner._index_map.f()[i] == site)
          named.push_back(i);
      REQUIRE(named.size() == 2);
      const auto standing = [site, survivor](int slot) {
        return slot == site ? survivor : slot;
      };
      dt3::resolve_tet_retired_sites(owner, standing);
      for (auto input : named)
        CHECK(owner._index_map.f()[input] == survivor);
      CHECK(collapse_volume6(owner) == volume);
      dt3::compact_tetrahedralization<false, true>(owner);
      CHECK(dt3::check_tetrahedralization_validity(
          owner, [site](int slot) { return slot == site; }));
      CHECK(collapse_volume6(owner) == volume);
      break;
    }
  }
  CHECK(collapsed == 1);
  CHECK(refused > 0);
}

TEST_CASE("collapse_tet_site: a hull site and a survivor off its star refuse",
          "[topology][delaunay3][collapse]") {
  collapse_owner owner;
  collapse_built(23u, 30, owner);
  const auto before = collapse_snapshot_of(owner);
  // a hull site refuses onto every neighbour
  int hull_site = -1;
  for (int site = 0; site < int(dt3::tet_site_count(owner)) && hull_site < 0;
       ++site) {
    bool hull = false;
    collapse_star_of(owner, site, hull);
    if (hull)
      hull_site = site;
  }
  REQUIRE(hull_site >= 0);
  bool hull_flag = false;
  for (auto c : collapse_star_of(owner, hull_site, hull_flag))
    for (auto corner : owner._corners[std::size_t(c)])
      if (corner != hull_site && corner != collapse_owner::infinite)
        CHECK_FALSE(dt3::collapse_tet_site(owner, hull_site, corner));
  CHECK(collapse_snapshot_of(owner) == before);
  // an interior site and a site its star does not reach
  for (int site = 0; site < int(dt3::tet_site_count(owner)); ++site) {
    bool hull = false;
    const auto star = collapse_star_of(owner, site, hull);
    if (hull)
      continue;
    std::vector<char> reached(dt3::tet_site_count(owner), 0);
    for (auto c : star)
      for (auto corner : owner._corners[std::size_t(c)])
        reached[std::size_t(corner)] = 1;
    int unreached = -1;
    for (int other = 0; other < int(reached.size()) && unreached < 0; ++other)
      if (!reached[std::size_t(other)])
        unreached = other;
    if (unreached < 0)
      continue;
    CHECK_FALSE(dt3::collapse_tet_site(owner, site, unreached));
    CHECK(collapse_snapshot_of(owner) == before);
    break;
  }
  CHECK_FALSE(dt3::collapse_tet_site(owner, 1, 1));
}

TEST_CASE("collapse_tet_site: a held facet at the site moves onto the "
          "survivor and names its face",
          "[topology][delaunay3][collapse]") {
  collapse_owner owner;
  collapse_built(31u, 40, owner);
  for (int site = 0; site < int(dt3::tet_site_count(owner)); ++site) {
    bool hull = false;
    const auto star = collapse_star_of(owner, site, hull);
    if (hull)
      continue;
    bool done = false;
    for (auto cell : star) {
      for (auto survivor : owner._corners[std::size_t(cell)]) {
        if (survivor == site)
          continue;
        // hold a facet of the star at the site, off the survivor
        int held_cell = -1;
        std::size_t held_slot = 0;
        for (auto c : star) {
          const auto corners = owner._corners[std::size_t(c)];
          if (std::find(corners.begin(), corners.end(), survivor) !=
              corners.end())
            continue;
          for (std::size_t k = 0; k < 4 && held_cell < 0; ++k)
            if (corners[k] != site) {
              const auto facet = dt3::tet_facet(corners, k);
              if (std::find(facet.begin(), facet.end(), survivor) ==
                  facet.end()) {
                held_cell = c;
                held_slot = k;
              }
            }
          if (held_cell >= 0)
            break;
        }
        if (held_cell < 0)
          continue;
        dt3::state_tet_facet_protection(owner, held_cell, held_slot, true, 7);
        REQUIRE(owner._n_held == 1);
        if (!dt3::collapse_tet_site(owner, site, survivor)) {
          dt3::state_tet_facet_protection(owner, held_cell, held_slot, false,
                                          collapse_owner::none);
          continue;
        }
        CHECK(owner._n_held == 1);
        REQUIRE(owner._collapse.faces.size() == 1);
        CHECK(owner._collapse.faces[0] == 7);
        CHECK(dt3::tet_facet_is_protected(owner, held_cell, held_slot));
        CHECK(collapse_held_bits(owner) == 2 * owner._n_held);
        done = true;
        break;
      }
      if (done)
        break;
    }
    if (done) {
      SUCCEED();
      return;
    }
  }
  FAIL("no held facet collapsed");
}

namespace {

/// The rows of `segment` walked from its first row through each `next`: the
/// sites in order, and whether the walk ends at the segment's last row.
auto collapse_chain_of(const collapse_recovery &recovery, std::size_t segment,
                       bool &ends_at_last) -> std::vector<int> {
  std::vector<int> sites;
  int k = recovery.first[segment], at = -1;
  for (; k != -1; k = recovery.subsegments[std::size_t(k)].next) {
    sites.push_back(recovery.subsegments[std::size_t(k)].a);
    at = k;
  }
  ends_at_last = at == recovery.last[segment];
  sites.push_back(recovery.subsegments[std::size_t(at)].b);
  return sites;
}

/// Whether `a`-`b` is an edge of a live finite cell.
auto collapse_is_edge(const collapse_owner &owner, int a, int b) -> bool {
  for (std::size_t c = 0; c < std::size_t(owner._corners.size()); ++c) {
    if (owner._states[c] == dt3::tet_state::dead)
      continue;
    const auto corners = owner._corners[c];
    if (std::find(corners.begin(), corners.end(), a) != corners.end() &&
        std::find(corners.begin(), corners.end(), b) != corners.end())
      return true;
  }
  return false;
}

} // namespace

TEST_CASE("collapse_tet_site_rows: the recovered rows at the site run to the "
          "survivor",
          "[topology][delaunay3][collapse]") {
  const int depth = 8;
  std::size_t joined = 0;
  for (std::uint32_t seed = 0; seed < 40 && !joined; ++seed) {
    CAPTURE(seed);
    // the site s stands on the segment a-b and on the segment c-d
    const std::int64_t unit = std::int64_t(1) << depth;
    std::vector<std::int64_t> flat{0,        0, 0, 4 * unit, 0,        0,
                                   unit,     0, 0, unit,     -2 * unit, 0,
                                   unit,     2 * unit, 0};
    std::mt19937 rng(9300 + seed);
    for (std::size_t i = 0; i < 3 * 40; ++i)
      flat.push_back(std::int64_t(tf::test::uniform_int(rng, -32, 32)) * unit);
    collapse_owner owner;
    REQUIRE(dt3::build_tetrahedralization(
        owner, tf::make_points<3>(tf::make_range(flat))));
    tf::none_t serial{};
    collapse_dt dt{owner, serial};
    const auto f = dt.index_map().f();
    const int a = f[0], b = f[1], s = f[2], c = f[3], d = f[4];
    const std::vector<std::array<int, 2>> segments{{a, b}, {c, d}};
    collapse_recovery recovery;
    dt3::compact_tet_dead_rows<true>(owner);
    REQUIRE(dt3::recover_tet_segments(dt, segments, 0, depth, recovery));
    dt3::compact_tetrahedralization<true, true>(owner);
    if (recovery.status[0] != dt3::tet_segment_status::recovered ||
        recovery.status[1] != dt3::tet_segment_status::recovered)
      continue;
    bool closed = false;
    const auto before0 = collapse_chain_of(recovery, 0, closed);
    const auto before1 = collapse_chain_of(recovery, 1, closed);
    // the survivor: the site before s on a-b, so that segment joins there
    const auto at = std::find(before0.begin(), before0.end(), s);
    REQUIRE(at != before0.end());
    const int survivor = *(at - 1);
    if (!dt3::collapse_tet_site_rows(owner, recovery, s, survivor))
      continue;
    ++joined;
    std::vector<int> expect0, expect1;
    for (auto v : before0)
      if (v != s)
        expect0.push_back(v);
    for (auto v : before1)
      expect1.push_back(v == s ? survivor : v);
    bool last0 = false, last1 = false;
    CHECK(collapse_chain_of(recovery, 0, last0) == expect0);
    CHECK(collapse_chain_of(recovery, 1, last1) == expect1);
    CHECK(last0);
    CHECK(last1);
    for (std::size_t i = 0; i + 1 < expect0.size(); ++i) {
      CHECK(dt3::tet_piece_row(recovery, expect0[i], expect0[i + 1]) != -1);
      CHECK(collapse_is_edge(owner, expect0[i], expect0[i + 1]));
    }
    for (std::size_t i = 0; i + 1 < expect1.size(); ++i) {
      CHECK(dt3::tet_piece_row(recovery, expect1[i], expect1[i + 1]) != -1);
      CHECK(collapse_is_edge(owner, expect1[i], expect1[i + 1]));
    }
    for (auto v : before0)
      CHECK(dt3::tet_piece_row(recovery, s, v) == -1);
    for (auto v : before1)
      CHECK(dt3::tet_piece_row(recovery, s, v) == -1);
    dt3::compact_tetrahedralization<false, true>(owner);
    CHECK(dt3::check_tetrahedralization_validity(
        owner, [s](int slot) { return slot == s; }));
  }
  CHECK(joined == 1);
}

TEST_CASE("collapse_tet_site: a held facet of a vanishing cell carries "
          "across the closure",
          "[topology][delaunay3][collapse]") {
  collapse_owner owner;
  collapse_built(41u, 40, owner);
  std::size_t carried = 0;
  for (int site = 0; site < int(dt3::tet_site_count(owner)) && !carried;
       ++site) {
    bool hull = false;
    const auto star = collapse_star_of(owner, site, hull);
    if (hull)
      continue;
    for (auto cell : star) {
      const auto corners = owner._corners[std::size_t(cell)];
      std::size_t is = 0;
      while (corners[is] != site)
        ++is;
      std::size_t iu = (is + 1) % 4;
      const int survivor = corners[iu];
      const int x = owner._neighbors[std::size_t(cell)][is];
      const int y = owner._neighbors[std::size_t(cell)][iu];
      // the facet holding the site and not the survivor, which closes onto
      // the one across from the site
      dt3::state_tet_facet_protection(owner, cell, iu, true, 7);
      REQUIRE(owner._n_held == 1);
      if (!dt3::collapse_tet_site(owner, site, survivor)) {
        dt3::state_tet_facet_protection(owner, cell, iu, false,
                                        collapse_owner::none);
        continue;
      }
      ++carried;
      CHECK(owner._states[std::size_t(cell)] == dt3::tet_state::dead);
      std::size_t bx = 0;
      while (bx < 4 && owner._neighbors[std::size_t(x)][bx] != y)
        ++bx;
      REQUIRE(bx < 4);
      std::size_t by = 0;
      while (by < 4 && owner._neighbors[std::size_t(y)][by] != x)
        ++by;
      REQUIRE(by < 4);
      CHECK(dt3::tet_facet_is_protected(owner, x, bx));
      CHECK(dt3::tet_facet_is_protected(owner, y, by));
      CHECK(owner._owners[4 * std::size_t(x) + bx] == 7);
      CHECK(owner._owners[4 * std::size_t(y) + by] == 7);
      CHECK(owner._n_held == 1);
      CHECK(collapse_held_bits(owner) == 2 * owner._n_held);
      REQUIRE(owner._collapse.faces.size() == 1);
      CHECK(owner._collapse.faces[0] == 7);
      dt3::compact_tetrahedralization<false, true>(owner);
      CHECK(dt3::check_tetrahedralization_validity(
          owner, [site](int slot) { return slot == site; }));
      break;
    }
  }
  CHECK(carried == 1);
}

TEST_CASE("collapse_tet_site: a held facet collapsing to no area names its "
          "face, and the face phase recovers it again",
          "[topology][delaunay3][collapse]") {
  using face_t = std::array<int, 3>;
  using phase_t = dt3::tet_segment_phase<int, std::int64_t>;
  using faces_t = dt3::tet_face_recovery<int, std::int64_t, tf::exact::int32>;
  const int depth = 12;
  const std::int64_t unit = std::int64_t(1) << depth;
  std::size_t recovered = 0;
  for (std::uint32_t seed = 0; seed < 20 && !recovered; ++seed) {
    CAPTURE(seed);
    // a triangle on z = 0 with three sites inside it, in a cloud off it
    std::vector<std::int64_t> flat{-60, -60, 0, 60,  -60, 0, 0, 60, 0,
                                   0,   -10, 0, 4,   0,   0, -6, 8, 0};
    std::mt19937 rng(9500 + seed);
    while (flat.size() < 3 * 46) {
      const std::int64_t p[3] = {tf::test::uniform_int(rng, -64, 64),
                                 tf::test::uniform_int(rng, -64, 64),
                                 tf::test::uniform_int(rng, -64, 64)};
      if (p[2] != 0)
        flat.insert(flat.end(), {p[0], p[1], p[2]});
    }
    for (auto &v : flat)
      v *= unit;
    collapse_owner owner;
    REQUIRE(dt3::build_tetrahedralization(
        owner, tf::make_points<3>(tf::make_range(flat))));
    const auto f = owner._index_map.f();
    const int a = f[0], b = f[1], c = f[2];
    std::vector<std::array<int, 2>> segments{{a, b}, {b, c}, {c, a}};
    tf::offset_block_buffer<int, face_t> facets;
    facets.offsets_buffer().allocate(2);
    facets.offsets_buffer()[0] = 0;
    facets.offsets_buffer()[1] = 1;
    facets.data_buffer().push_back(face_t{a, b, c});
    const auto blocks = tf::make_offset_block_range(facets.offsets_buffer(),
                                                    facets.data_buffer());
    phase_t phase;
    faces_t recovery;
    phase.segment.site_budget = dt3::tet_site_count(owner) * 32;
    tf::none_t serial{};
    const auto name = [](const auto &, auto &) { return false; };
    REQUIRE(dt3::recover_tet_segment_phase<false>(
        owner, serial, segments, blocks, tf::make_sequence_range(1), depth,
        phase, tf::none_t{}, name));
    dt3::compact_tetrahedralization<false, true>(owner);
    dt3::recover_tet_face_phase<false>(owner, serial, blocks,
                                       tf::make_sequence_range(1), phase,
                                       recovery);
    dt3::compact_tetrahedralization<false, true>(owner);
    if (recovery.status[0] != dt3::tet_face_status::recovered)
      continue;
    // an interior site of the face onto a neighbour on the face, along an
    // edge a held facet holds
    bool done = false;
    int retired_site = -1;
    for (int site : {f[3], f[4], f[5]}) {
      bool hull = false;
      const auto star = collapse_star_of(owner, site, hull);
      for (auto cell : star) {
        const auto corners = owner._corners[std::size_t(cell)];
        for (std::size_t k = 0; k < 4 && !done; ++k) {
          if (corners[k] == site ||
              !dt3::tet_facet_is_protected(owner, cell, k))
            continue;
          for (auto survivor : dt3::tet_facet(corners, k)) {
            if (survivor == site)
              continue;
            if (!dt3::collapse_tet_site_rows(owner, phase.segment, site,
                                             survivor))
              continue;
            retired_site = site;
            done = true;
            break;
          }
        }
        if (done)
          break;
      }
      if (done)
        break;
    }
    if (!done)
      continue;
    REQUIRE(owner._collapse.faces.size() == 1);
    CHECK(owner._collapse.faces[0] == 0);
    CHECK(collapse_held_bits(owner) == 2 * owner._n_held);
    dt3::compact_tetrahedralization<false, true>(owner);
    const std::vector<int> due(owner._collapse.faces.begin(),
                               owner._collapse.faces.end());
    dt3::recover_tet_face_phase<false>(owner, serial, blocks,
                                       tf::make_range(due), phase, recovery);
    dt3::compact_tetrahedralization<false, true>(owner);
    CHECK(recovery.status[0] == dt3::tet_face_status::recovered);
    CHECK(dt3::check_tetrahedralization_validity(
        owner, [retired_site](int slot) { return slot == retired_site; }));
    CHECK(dt3::check_tet_face_fixpoint(owner, blocks, recovery));
    ++recovered;
  }
  CHECK(recovered == 1);
}

TEST_CASE("collapse_tet_site_rows: a segment end collapses, and the surface "
          "restated on the survivor ends its segments there",
          "[topology][delaunay3][collapse]") {
  using face_t = std::array<int, 3>;
  using phase_t = dt3::tet_segment_phase<int, std::int64_t>;
  using faces_t = dt3::tet_face_recovery<int, std::int64_t, tf::exact::int32>;
  const int depth = 12;
  const std::int64_t unit = std::int64_t(1) << depth;
  std::size_t restated = 0;
  for (std::uint32_t seed = 0; seed < 20 && !restated; ++seed) {
    CAPTURE(seed);
    // a triangle on z = 0 with three sites inside it, in a cloud off it
    std::vector<std::int64_t> flat{-30, -30, 0, 30,  -30, 0, 0,  30, 0,
                                   -20, -22, 0, 2,   0,   0, -3, 4,  0};
    std::mt19937 rng(9700 + seed);
    while (flat.size() < 3 * 86) {
      const std::int64_t p[3] = {tf::test::uniform_int(rng, -64, 64),
                                 tf::test::uniform_int(rng, -64, 64),
                                 tf::test::uniform_int(rng, -64, 64)};
      if (p[2] != 0)
        flat.insert(flat.end(), {p[0], p[1], p[2]});
    }
    for (auto &v : flat)
      v *= unit;
    collapse_owner owner;
    REQUIRE(dt3::build_tetrahedralization(
        owner, tf::make_points<3>(tf::make_range(flat))));
    const auto f = owner._index_map.f();
    const int a = f[0], b = f[1], c = f[2];
    tf::offset_block_buffer<int, face_t> facets;
    facets.offsets_buffer().allocate(2);
    facets.offsets_buffer()[0] = 0;
    facets.offsets_buffer()[1] = 1;
    facets.data_buffer().push_back(face_t{a, b, c});
    phase_t phase;
    faces_t recovery;
    phase.segment.site_budget = dt3::tet_site_count(owner) * 32;
    tf::none_t serial{};
    const auto name = [](const auto &, auto &) { return false; };
    const std::vector<std::array<int, 2>> free_segments;
    {
      const auto blocks = tf::make_offset_block_range(facets.offsets_buffer(),
                                                      facets.data_buffer());
      REQUIRE(dt3::recover_tet_segment_phase<false>(
          owner, serial, free_segments, blocks, tf::make_sequence_range(1),
          depth, phase, tf::none_t{}, name));
      dt3::compact_tetrahedralization<false, true>(owner);
      dt3::recover_tet_face_phase<false>(owner, serial, blocks,
                                         tf::make_sequence_range(1), phase,
                                         recovery);
      dt3::compact_tetrahedralization<false, true>(owner);
    }
    if (recovery.status[0] != dt3::tet_face_status::recovered)
      continue;
    // the corner a onto a site inside the face its star holds
    int survivor = -1;
    for (int u : {f[3], f[4], f[5]})
      if (dt3::collapse_tet_site_rows(owner, phase.segment, a, u)) {
        survivor = u;
        break;
      }
    if (survivor < 0)
      continue;
    ++restated;
    std::vector<int> ending;
    for (std::size_t s = 0; s < phase.segments.size(); ++s)
      if (phase.segments[s][0] == a || phase.segments[s][1] == a)
        ending.push_back(int(s));
    REQUIRE(ending.size() == 2);
    // the face holding the corner retires and is stated on the survivor
    recovery.status[0] = dt3::tet_face_status::retired;
    tf::buffer<int> retired;
    retired.push_back(0);
    tf::buffer<dt3::tet_retired_facet<int>> retiring;
    dt3::retire_tet_faces(owner, retired, dt3::tet_face_rim_sites(phase),
                          retiring);
    tf::buffer<int> candidates, held;
    dt3::retire_tet_segments(phase, recovery.status, retired, candidates,
                             held);
    for (auto s : ending)
      CHECK(phase.segment.status[std::size_t(s)] ==
            dt3::tet_segment_status::retired);
    facets.offsets_buffer().push_back(2);
    facets.data_buffer().push_back(face_t{survivor, b, c});
    const auto blocks = tf::make_offset_block_range(facets.offsets_buffer(),
                                                    facets.data_buffer());
    const std::vector<int> due{1};
    REQUIRE(dt3::recover_tet_segment_phase<false>(
        owner, serial, free_segments, blocks, tf::make_range(due), depth,
        phase, tf::none_t{}, name));
    dt3::compact_tetrahedralization<false, true>(owner);
    dt3::recover_tet_face_phase<false>(owner, serial, blocks,
                                       tf::make_range(due), phase, recovery);
    dt3::compact_tetrahedralization<false, true>(owner);
    CHECK(recovery.status[1] == dt3::tet_face_status::recovered);
    // the survivor's segments: chains from end to end, the ends' numerators
    // the frame's, every piece found through the rows at its lower site
    std::size_t through = 0;
    for (auto s : phase.edges[std::size_t(phase.edge_block[1])]) {
      const auto &rec = phase.segment;
      CHECK(rec.status[std::size_t(s)] == dt3::tet_segment_status::recovered);
      bool at_last = false;
      const auto chain = collapse_chain_of(rec, std::size_t(s), at_last);
      CHECK(at_last);
      const auto ends = phase.segments[std::size_t(s)];
      CHECK(std::min(chain.front(), chain.back()) == ends[0]);
      CHECK(std::max(chain.front(), chain.back()) == ends[1]);
      CHECK(rec.subsegments[std::size_t(rec.first[std::size_t(s)])].n0 == 0);
      CHECK(rec.subsegments[std::size_t(rec.last[std::size_t(s)])].n1 ==
            std::int64_t(1) << rec.bits[std::size_t(s)]);
      for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
        CHECK(dt3::tet_piece_row(rec, chain[i], chain[i + 1]) != -1);
        CHECK(collapse_is_edge(owner, chain[i], chain[i + 1]));
      }
      through += std::size_t(ends[0] == survivor || ends[1] == survivor);
    }
    CHECK(through == 2);
    for (std::size_t s = 0; s < phase.segments.size(); ++s)
      if (phase.segment.status[s] != dt3::tet_segment_status::retired)
        CHECK((phase.segments[s][0] != a && phase.segments[s][1] != a));
    CHECK(collapse_held_bits(owner) == 2 * owner._n_held);
    CHECK(dt3::check_tetrahedralization_validity(
        owner, [a](int slot) { return slot == a; }));
    CHECK(dt3::check_tet_face_fixpoint(owner, blocks, recovery));
  }
  CHECK(restated == 1);
}
