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
#include "worker_ladder.hpp"
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/tetrahedra_buffer.hpp>
#include <trueform/core/tetras.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/exact/insphere.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/orient3d.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/cdt/dt3/append_tetrahedralization_sites.hpp>
#include <trueform/topology/cdt/dt3/assign_tet_compaction.hpp>
#include <trueform/topology/cdt/dt3/build_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/check_tetrahedralization_validity.hpp>
#include <trueform/topology/cdt/dt3/clear_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/clear_tetrahedralization_cells.hpp>
#include <trueform/topology/cdt/dt3/close_tet_cavity_links.hpp>
#include <trueform/topology/cdt/dt3/compact_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/insert_tetrahedralization_site.hpp>
#include <trueform/topology/cdt/dt3/load_tetrahedralization_sites.hpp>
#include <trueform/topology/cdt/dt3/measure_tet_site_domain.hpp>
#include <trueform/topology/cdt/dt3/open_tet_cavity_links.hpp>
#include <trueform/topology/cdt/dt3/order_tetrahedralization_sites.hpp>
#include <trueform/topology/cdt/dt3/prepare_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/protect_tet_facets.hpp>
#include <trueform/topology/cdt/dt3/seed_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/state_tet_cavity_facet.hpp>
#include <trueform/topology/cdt/dt3/tet_cavity_links.hpp>
#include <trueform/topology/cdt/dt3/tet_claim_workspace.hpp>
#include <trueform/topology/cdt/dt3/tet_conflicts.hpp>
#include <trueform/topology/cdt/dt3/tet_execution_tuning.hpp>
#include <trueform/topology/cdt/dt3/tet_facets.hpp>
#include <trueform/topology/cdt/dt3/tet_insertion.hpp>
#include <trueform/topology/cdt/dt3/tet_protected_count.hpp>
#include <trueform/topology/cdt/dt3/tet_site_level.hpp>
#include <trueform/topology/cdt/dt3/tet_site_morton_key.hpp>
#include <trueform/topology/cdt/dt3/tet_site_position.hpp>
#include <trueform/topology/cdt/dt3/tetrahedralization_owner.hpp>
#include <trueform/topology/cdt/dt3/weld_tetrahedralization_sites.hpp>
#include <trueform/topology/delaunay_tetrahedralizer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <tbb/task_arena.h>
#include <type_traits>
#include <vector>

namespace {

using dt3_i32 = tf::exact::int32;
using dt3_i64 = tf::exact::int64;
using dt3_scaled_coord = tf::exact::meta<dt3_i32>::T1;

template <typename Int = dt3_i32, typename Coord = Int>
using dt3_builder = tf::delaunay_tetrahedralizer<int, Coord, Int>;

template <typename Coord>
auto dt3_lattice_points(const std::vector<Coord> &flat) {
  return tf::make_points<3>(tf::make_range(flat));
}

template <typename Coord>
auto dt3_random_coordinates(std::uint32_t seed, std::size_t sites, Coord span)
    -> std::vector<Coord> {
  std::mt19937 rng(seed);
  const auto reach = (long long)(span);
  std::vector<Coord> flat;
  flat.reserve(sites * 3);
  for (std::size_t i = 0; i < sites * 3; ++i)
    flat.push_back(Coord(tf::test::uniform_int(rng, -reach, reach)));
  return flat;
}

/// The corner quadruples, each sorted and the whole set sorted: the identity
/// two runs, or a run and a reference, are compared on.
template <typename Builder>
auto dt3_canonical_cells(const Builder &dt) -> std::vector<std::array<int, 4>> {
  std::vector<std::array<int, 4>> cells;
  for (auto cell : dt.tets()) {
    std::array<int, 4> key{int(cell[0]), int(cell[1]), int(cell[2]),
                           int(cell[3])};
    std::sort(key.begin(), key.end());
    cells.push_back(key);
  }
  std::sort(cells.begin(), cells.end());
  return cells;
}

template <typename View>
auto dt3_named_cells(const View &view) -> std::vector<std::array<int, 4>> {
  auto cells = dt3_canonical_cells(view);
  for (auto &cell : cells) {
    for (auto &site : cell)
      site = view.sites()[std::size_t(site)].id;
    std::sort(cell.begin(), cell.end());
  }
  std::sort(cells.begin(), cells.end());
  return cells;
}

using dt3_owner =
    tf::topology::cdt::dt3::tetrahedralization_owner<int, dt3_i32, dt3_i32>;

/// The same quadruples read off an owner the test drove itself.
auto dt3_owner_cells(const dt3_owner &owner)
    -> std::vector<std::array<int, 4>> {
  std::vector<std::array<int, 4>> cells;
  for (std::size_t c = 0; c < owner._n_finite; ++c) {
    std::array<int, 4> key{owner._corners[c][0], owner._corners[c][1],
                           owner._corners[c][2], owner._corners[c][3]};
    std::sort(key.begin(), key.end());
    cells.push_back(key);
  }
  std::sort(cells.begin(), cells.end());
  return cells;
}

/// The library's own build with one hook: the order lane is permuted after
/// the ordering states it, so the same welded sites and the same canonical
/// names are inserted on another schedule.
template <typename Sites, typename Permute>
auto dt3_build_in_schedule(dt3_owner &owner, const Sites &sites,
                           const Permute &permute) -> bool {
  namespace dt3 = tf::topology::cdt::dt3;
  dt3::clear_tetrahedralization(owner);
  dt3::load_tetrahedralization_sites(owner, sites);
  dt3::weld_tetrahedralization_sites(owner);
  if (!dt3::seed_tetrahedralization(owner))
    return false;
  const std::array<int, 4> seeded{owner._corners[0][0], owner._corners[0][1],
                                  owner._corners[0][2], owner._corners[0][3]};
  dt3::order_tetrahedralization_sites(owner);
  permute(owner._order);
  for (std::size_t i = 0; i < owner._order.size(); ++i) {
    const int site = owner._order[i];
    if (site == seeded[0] || site == seeded[1] || site == seeded[2] ||
        site == seeded[3])
      continue;
    if (dt3::insert_tetrahedralization_site(owner, site) !=
        dt3::tet_insertion::placed)
      return false;
  }
  dt3::compact_tetrahedralization(owner);
  return true;
}

/// No site inside any finite cell's circumsphere, by the same symbolic
/// tie-break the kernel decided with.
template <typename Int, typename Builder>
auto dt3_spheres_are_empty(const Builder &dt) -> bool {
  const auto sites = dt.sites();
  for (auto cell : dt.tets())
    for (std::size_t s = 0; s < sites.size(); ++s) {
      if (std::size_t(cell[0]) == s || std::size_t(cell[1]) == s ||
          std::size_t(cell[2]) == s || std::size_t(cell[3]) == s)
        continue;
      const std::array<typename Builder::site_type, 5> vs{
          sites[std::size_t(cell[0])], sites[std::size_t(cell[1])],
          sites[std::size_t(cell[2])], sites[std::size_t(cell[3])], sites[s]};
      if (tf::exact::insphere_conflict_scaled<Int>(vs.data(), 1))
        return false;
    }
  return true;
}

/// The published product read on its own terms: every finite cell wound
/// positively, every facet carried by two finite cells or by one and a hull
/// slot, every neighbour naming back, every site a corner, and the hull
/// facets closing a surface.
template <typename Int, typename Builder>
auto dt3_product_holds(const Builder &dt) -> bool {
  struct facet {
    int a;
    int b;
    int c;
    int cell;
    int slot;
  };
  const auto sites = dt.sites();
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  if (cells.size() == 0)
    return false;

  std::vector<facet> facets;
  std::vector<std::array<int, 2>> hull;
  std::vector<char> covered(sites.size(), 0);
  for (std::size_t c = 0; c < std::size_t(cells.size()); ++c) {
    const auto cell = cells[c];
    if (!(tf::exact::orient3d_value_scaled<Int>(
              sites[std::size_t(cell[0])].pt, sites[std::size_t(cell[1])].pt,
              sites[std::size_t(cell[2])].pt,
              sites[std::size_t(cell[3])].pt) > 0))
      return false;
    for (std::size_t k = 0; k < 4; ++k)
      covered[std::size_t(cell[k])] = 1;
    for (std::size_t slot = 0; slot < 4; ++slot) {
      auto corners = tf::topology::cdt::dt3::tet_facet(cell, slot);
      if (neighbors[c][slot] == Builder::k_none) {
        hull.push_back({corners[0], corners[1]});
        hull.push_back({corners[1], corners[2]});
        hull.push_back({corners[2], corners[0]});
      }
      std::sort(corners.begin(), corners.end());
      facets.push_back({corners[0], corners[1], corners[2], int(c), int(slot)});
    }
  }
  for (std::size_t i = 0; i < covered.size(); ++i)
    if (!covered[i])
      return false;

  std::sort(facets.begin(), facets.end(), [](const facet &x, const facet &y) {
    return x.a < y.a ||
           (x.a == y.a && (x.b < y.b || (x.b == y.b && x.c < y.c)));
  });
  for (std::size_t i = 0; i < facets.size();) {
    std::size_t j = i;
    while (j < facets.size() && facets[j].a == facets[i].a &&
           facets[j].b == facets[i].b && facets[j].c == facets[i].c)
      ++j;
    if (j - i == 2) {
      if (neighbors[std::size_t(facets[i].cell)][std::size_t(facets[i].slot)] !=
              facets[i + 1].cell ||
          neighbors[std::size_t(facets[i + 1].cell)]
                   [std::size_t(facets[i + 1].slot)] != facets[i].cell)
        return false;
    } else if (j - i == 1) {
      if (neighbors[std::size_t(facets[i].cell)][std::size_t(facets[i].slot)] !=
          Builder::k_none)
        return false;
    } else
      return false;
    i = j;
  }

  std::sort(hull.begin(), hull.end());
  if (std::adjacent_find(hull.begin(), hull.end()) != hull.end())
    return false;
  for (auto &edge : hull)
    if (edge[1] < edge[0])
      std::swap(edge[0], edge[1]);
  std::sort(hull.begin(), hull.end());
  if (hull.empty() || hull.size() % 2 != 0)
    return false;
  for (std::size_t i = 0; i < hull.size(); i += 2)
    if (hull[i] != hull[i + 1] ||
        (i + 2 < hull.size() && hull[i + 2] == hull[i]))
      return false;
  return true;
}

/// Every four-subset that is orientable and whose circumsphere admits no
/// other site: the Delaunay complex by definition, at O(n^4).
template <typename Int, typename Builder>
auto dt3_brute_force_cells(const Builder &dt)
    -> std::vector<std::array<int, 4>> {
  const auto sites = dt.sites();
  const int n = int(sites.size());
  std::vector<std::array<int, 4>> cells;
  for (int a = 0; a < n; ++a)
    for (int b = a + 1; b < n; ++b)
      for (int c = b + 1; c < n; ++c)
        for (int d = c + 1; d < n; ++d) {
          const int orientation = tf::exact::orient3d_sign<Int>(
              sites[std::size_t(a)].pt, sites[std::size_t(b)].pt,
              sites[std::size_t(c)].pt, sites[std::size_t(d)].pt);
          if (orientation == 0)
            continue;
          bool empty = true;
          for (int e = 0; e < n && empty; ++e) {
            if (e == a || e == b || e == c || e == d)
              continue;
            const std::array<typename Builder::site_type, 5> vs{
                sites[std::size_t(a)], sites[std::size_t(b)],
                sites[std::size_t(c)], sites[std::size_t(d)],
                sites[std::size_t(e)]};
            empty = !tf::exact::insphere_conflict_scaled<Int>(vs.data(),
                                                              orientation);
          }
          if (empty)
            cells.push_back({a, b, c, d});
        }
  std::sort(cells.begin(), cells.end());
  return cells;
}

} // namespace

TEST_CASE("delaunay tets: random lattice sites build a valid complex",
          "[topology][delaunay3]") {
  for (std::size_t n : {std::size_t(16), std::size_t(60), std::size_t(200)}) {
    dt3_builder<> dt;
    const auto flat =
        dt3_random_coordinates<dt3_i32>(std::uint32_t(1000 + n), n, 4000);
    REQUIRE(dt.build(dt3_lattice_points(flat)));
    CHECK(dt.refusal() == tf::tetrahedralization_refusal::none);
    CHECK(dt.n_sites() == n);
    CHECK(dt.n_tets() > 0);
    CHECK(dt.is_valid());
    CHECK(dt3_product_holds<dt3_i32>(dt));
    CHECK(dt3_spheres_are_empty<dt3_i32>(dt));
  }

  dt3_builder<dt3_i64> wide;
  const auto flat = dt3_random_coordinates<dt3_i64>(77u, 120, 1000000000LL);
  REQUIRE(wide.build(dt3_lattice_points(flat)));
  CHECK(wide.is_valid());
  CHECK(dt3_product_holds<dt3_i64>(wide));
  CHECK(dt3_spheres_are_empty<dt3_i64>(wide));
}

TEST_CASE("delaunay tets: the cells are the brute-force Delaunay set",
          "[.fuzz][topology][delaunay3]") {
  int compared = 0;
  for (std::uint32_t seed = 0; seed < 12; ++seed) {
    dt3_builder<> dt;
    const auto flat =
        dt3_random_coordinates<dt3_i32>(seed, 6 + std::size_t(seed % 7), 40);
    if (!dt.build(dt3_lattice_points(flat)))
      continue;
    REQUIRE(dt.is_valid());
    const auto built = dt3_canonical_cells(dt);
    REQUIRE(!built.empty());
    CHECK(built == dt3_brute_force_cells<dt3_i32>(dt));
    ++compared;
  }
  CHECK(compared == 12);
}

TEST_CASE("delaunay tets: a cospherical grid stores no flat cell",
          "[topology][delaunay3]") {
  for (int side = 2; side <= 4; ++side) {
    std::vector<dt3_i32> flat;
    for (int x = 0; x < side; ++x)
      for (int y = 0; y < side; ++y)
        for (int z = 0; z < side; ++z) {
          flat.push_back(x * 4);
          flat.push_back(y * 4);
          flat.push_back(z * 4);
        }
    dt3_builder<> dt;
    REQUIRE(dt.build(dt3_lattice_points(flat)));
    CHECK(dt.n_sites() == std::size_t(side * side * side));
    // Every cell of the grid falls to six tetrahedra, the same way in every
    // cell: nothing about the ties is decided by where the cell sits.
    CHECK(dt.n_tets() == std::size_t(6 * (side - 1) * (side - 1) * (side - 1)));
    CHECK(dt.is_valid());
    CHECK(dt3_product_holds<dt3_i32>(dt));
    CHECK(dt3_spheres_are_empty<dt3_i32>(dt));
  }
}

TEST_CASE("delaunay tets: the seed scan bypasses collinear and coplanar runs",
          "[topology][delaunay3]") {
  // The first four sites in canonical order are collinear, the next three
  // stay in their plane, and only the last leaves it.
  const std::vector<dt3_i32> flat = {0, 0, 0, 1, 0, 0, 2, 0, 0, 3,  0, 0,
                                     0, 1, 0, 0, 2, 0, 1, 1, 0, -5, 3, 7};
  dt3_builder<> dt;
  REQUIRE(dt.build(dt3_lattice_points(flat)));
  CHECK(dt.n_sites() == 8);
  CHECK(dt.is_valid());
  CHECK(dt3_product_holds<dt3_i32>(dt));
  CHECK(dt3_spheres_are_empty<dt3_i32>(dt));
}

TEST_CASE("delaunay tets: rank below three refuses", "[topology][delaunay3]") {
  dt3_builder<> planar;
  const std::vector<dt3_i32> plane = {0, 0, 0, 4, 0, 0, 0, 4,
                                      0, 4, 4, 0, 1, 2, 0};
  CHECK(!planar.build(dt3_lattice_points(plane)));
  CHECK(planar.refusal() == tf::tetrahedralization_refusal::rank_deficient);
  CHECK(planar.n_tets() == 0);
  CHECK(!planar.is_valid());
  // Preparation stands on its own, so the sites it welded survive.
  CHECK(planar.n_sites() == 5);

  dt3_builder<> collinear;
  const std::vector<dt3_i32> line = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  CHECK(!collinear.build(dt3_lattice_points(line)));
  CHECK(collinear.refusal() == tf::tetrahedralization_refusal::rank_deficient);

  dt3_builder<> sparse;
  const std::vector<dt3_i32> three = {0, 0, 0, 4, 0, 0, 0, 4, 0};
  CHECK(!sparse.build(dt3_lattice_points(three)));
  CHECK(sparse.refusal() == tf::tetrahedralization_refusal::rank_deficient);
}

TEST_CASE("delaunay tets: duplicate positions weld to one site",
          "[topology][delaunay3]") {
  const std::vector<dt3_i32> flat = {0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 8,
                                     0, 0, 0, 8, 8, 0, 0, 3, 3, 3};
  dt3_builder<> dt;
  REQUIRE(dt.build(dt3_lattice_points(flat)));
  CHECK(dt.n_sites() == 5);
  const auto &map = dt.index_map();
  CHECK(map.f()[0] == map.f()[1]);
  CHECK(map.f()[2] == map.f()[5]);
  CHECK(map.kept_ids()[std::size_t(map.f()[0])] == 0);
  CHECK(map.kept_ids()[std::size_t(map.f()[2])] == 2);
  CHECK(dt.is_valid());
  CHECK(dt3_product_holds<dt3_i32>(dt));
}

TEST_CASE("delaunay tets: a site in a hull face's plane inserts",
          "[topology][delaunay3]") {
  // The tilted witness: p shares the plane of face abc, so the hull cell
  // over it sees nothing and must ask its finite mate instead.
  const tf::exact::pt3<dt3_i32> a{0, 0, 0}, b{4, 0, 4}, c{0, 4, 0}, d{0, 0, -4},
      p{3, 5, 3};
  REQUIRE(tf::exact::orient3d_sign<dt3_i32>(a, b, c, p) == 0);
  REQUIRE(tf::exact::orient3d_sign<dt3_i32>(a, b, c, d) != 0);

  const std::vector<dt3_i32> flat = {a[0], a[1], a[2], b[0], b[1],
                                     b[2], c[0], c[1], c[2], d[0],
                                     d[1], d[2], p[0], p[1], p[2]};
  dt3_builder<> dt;
  REQUIRE(dt.build(dt3_lattice_points(flat)));
  CHECK(dt.n_sites() == 5);
  CHECK(dt.n_tets() > 0);
  CHECK(dt.is_valid());
  CHECK(dt3_product_holds<dt3_i32>(dt));
  CHECK(dt3_spheres_are_empty<dt3_i32>(dt));
}

TEST_CASE("delaunay tets: the hull-edge verdict is the edge's alone",
          "[topology][delaunay3]") {
  // On the line through a hull edge the normalized comparison collapses to
  // t(1-t)|b-a|^2, so no third vertex and no apex can change it. Here
  // |b-a|^2 = 64, and the queries at t = -1, 1/4, 1/2, 3/4, 2 read -128, 12,
  // 16, 12, -128.
  const tf::exact::pt3<dt3_i32> a{0, 0, 0}, b{8, 0, 0};
  const std::array<tf::exact::pt3<dt3_i32>, 5> queries{
      tf::exact::pt3<dt3_i32>{-8, 0, 0}, tf::exact::pt3<dt3_i32>{2, 0, 0},
      tf::exact::pt3<dt3_i32>{4, 0, 0}, tf::exact::pt3<dt3_i32>{6, 0, 0},
      tf::exact::pt3<dt3_i32>{16, 0, 0}};
  const std::array<int, 5> constants{-128, 12, 16, 12, -128};
  const std::array<tf::exact::pt3<dt3_i32>, 3> thirds{
      tf::exact::pt3<dt3_i32>{0, 5, 0}, tf::exact::pt3<dt3_i32>{3, -7, 0},
      tf::exact::pt3<dt3_i32>{-2, 9, 0}};
  const std::array<tf::exact::pt3<dt3_i32>, 3> apexes{
      tf::exact::pt3<dt3_i32>{1, 1, 6}, tf::exact::pt3<dt3_i32>{4, 2, -9},
      tf::exact::pt3<dt3_i32>{-3, 3, 5}};

  int checked = 0;
  for (std::size_t q = 0; q < queries.size(); ++q)
    for (const auto &c : thirds)
      for (const auto &u : apexes) {
        const auto orientation = tf::exact::orient3d_value<dt3_i32>(a, b, c, u);
        REQUIRE(orientation != 0);
        using lane = tf::exact::meta<dt3_i32>::T3;
        REQUIRE(tf::exact::insphere_value<dt3_i32>(a, b, c, u, queries[q]) ==
                lane(orientation) * lane(constants[q]));
        const std::array<tf::exact::vertex<int, dt3_i32>, 5> vs{
            tf::exact::vertex<int, dt3_i32>{0, a},
            tf::exact::vertex<int, dt3_i32>{1, b},
            tf::exact::vertex<int, dt3_i32>{2, c},
            tf::exact::vertex<int, dt3_i32>{3, u},
            tf::exact::vertex<int, dt3_i32>{4, queries[q]}};
        const int sign = orientation > 0 ? 1 : -1;
        CHECK(tf::exact::insphere_conflict(vs, sign) == (constants[q] > 0));
        ++checked;
      }
  CHECK(checked == 45);
}

TEST_CASE("delaunay tets: a site on a hull edge's line inserts",
          "[topology][delaunay3]") {
  for (int along : {-8, 2, 4, 6, 16}) {
    const std::vector<dt3_i32> flat = {
        0, 0, 0, 8, 0, 0, 0, 8, 0, 0, 0, 8, -4, 4, 4, dt3_i32(along), 0, 0};
    dt3_builder<> dt;
    REQUIRE(dt.build(dt3_lattice_points(flat)));
    CHECK(dt.n_sites() == 6);
    CHECK(dt.is_valid());
    CHECK(dt3_product_holds<dt3_i32>(dt));
    CHECK(dt3_spheres_are_empty<dt3_i32>(dt));
  }
}

TEST_CASE("delaunay tets: repeated builds give the same cells",
          "[topology][delaunay3]") {
  const auto flat = dt3_random_coordinates<dt3_i32>(4242u, 250, 500);
  dt3_builder<> first;
  dt3_builder<> second;
  REQUIRE(first.build(dt3_lattice_points(flat)));
  REQUIRE(second.build(dt3_lattice_points(flat)));
  CHECK(dt3_canonical_cells(first) == dt3_canonical_cells(second));
  // The same instance reused reaches the same complex as a fresh one.
  REQUIRE(first.build(dt3_lattice_points(flat)));
  CHECK(dt3_canonical_cells(first) == dt3_canonical_cells(second));
  CHECK(first.stats().insertions == second.stats().insertions);
  CHECK(first.stats().compactions == second.stats().compactions);
  // A serial build's star takes its cavity's rows, so no dead slot is left
  // to sweep and the one compaction is the publication.
  CHECK(first.stats().compactions == 1);
}

TEST_CASE("delaunay tets: the cells refuse to outgrow the index",
          "[topology][delaunay3]") {
  tf::delaunay_tetrahedralizer<std::int8_t, dt3_i32> tiny;
  const auto flat = dt3_random_coordinates<dt3_i32>(9u, 40, 100);
  CHECK(!tiny.build(dt3_lattice_points(flat)));
  CHECK(tiny.refusal() == tf::tetrahedralization_refusal::index_capacity);
  CHECK(tiny.n_tets() == 0);
  CHECK(!tiny.is_valid());
}

TEST_CASE("delaunay tets: the entry shapes", "[topology][delaunay3]") {
  const auto flat = dt3_random_coordinates<dt3_i32>(31u, 40, 200);
  dt3_builder<> by_points;
  REQUIRE(by_points.build(dt3_lattice_points(flat)));

  std::vector<tf::exact::vertex<int, dt3_i32>> carried;
  for (std::size_t i = 0; i < flat.size() / 3; ++i)
    carried.push_back(
        {int(i), tf::exact::pt3<dt3_i32>{flat[i * 3], flat[i * 3 + 1],
                                         flat[i * 3 + 2]}});
  const auto &frozen = carried;
  dt3_builder<> by_sites;
  REQUIRE(by_sites.build(tf::make_range(frozen)));
  // The carried names are the slots the points entry gives itself, so the
  // two entries state the same complex.
  CHECK(dt3_canonical_cells(by_points) == dt3_canonical_cells(by_sites));

  std::vector<dt3_scaled_coord> scaled;
  for (auto v : flat)
    scaled.push_back(dt3_scaled_coord(v) * 64);
  dt3_builder<dt3_i32, dt3_scaled_coord> by_scale;
  REQUIRE(by_scale.build(dt3_lattice_points(scaled)));
  CHECK(by_scale.is_valid());
  // A uniform positive scale multiplies every predicate by a positive power
  // of itself, so the complex is the unscaled one by name; the sites are
  // stored in the Morton order of their own domain, which the scale moves.
  CHECK(dt3_named_cells(by_scale) == dt3_named_cells(by_points));
}

TEST_CASE("delaunay tets: the validity gate rejects a broken complex",
          "[topology][delaunay3]") {
  tf::topology::cdt::dt3::tetrahedralization_owner<int, dt3_i32, dt3_i32> owner;
  const auto flat = dt3_random_coordinates<dt3_i32>(5u, 24, 300);
  REQUIRE(tf::topology::cdt::dt3::build_tetrahedralization(
      owner, dt3_lattice_points(flat)));
  REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));

  const auto swapped = owner._corners[0][1];
  owner._corners[0][1] = owner._corners[0][2];
  owner._corners[0][2] = swapped;
  CHECK(!tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
  owner._corners[0][2] = owner._corners[0][1];
  owner._corners[0][1] = swapped;
  REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));

  const auto ticket = owner._neighbors[0][0];
  owner._neighbors[0][0] = owner._neighbors[0][1];
  CHECK(!tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
  owner._neighbors[0][0] = ticket;
  REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));

  owner._n_finite -= 1;
  CHECK(!tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
}

TEST_CASE("delaunay tets: the cells are the same on any insertion schedule",
          "[topology][delaunay3]") {
  std::mt19937 rng(20260922u);
  std::vector<std::vector<dt3_i32>> cases;
  for (std::uint32_t seed = 0; seed < 4; ++seed)
    cases.push_back(dt3_random_coordinates<dt3_i32>(
        seed, 40 + 60 * std::size_t(seed), 900));
  // The cospherical grid is where a schedule could move a tie if any did.
  std::vector<dt3_i32> grid;
  for (int x = 0; x < 4; ++x)
    for (int y = 0; y < 4; ++y)
      for (int z = 0; z < 4; ++z) {
        grid.push_back(x * 4);
        grid.push_back(y * 4);
        grid.push_back(z * 4);
      }
  cases.push_back(grid);

  for (const auto &flat : cases) {
    const auto points = dt3_lattice_points(flat);
    dt3_owner pinned;
    REQUIRE(dt3_build_in_schedule(pinned, points, [](tf::buffer<int> &) {}));
    const auto reference = dt3_owner_cells(pinned);
    REQUIRE(!reference.empty());

    // The hook reproduces the library's own build when it permutes nothing.
    dt3_builder<> published;
    REQUIRE(published.build(points));
    CHECK(dt3_canonical_cells(published) == reference);

    dt3_owner ascending;
    REQUIRE(
        dt3_build_in_schedule(ascending, points, [](tf::buffer<int> &order) {
          std::iota(order.begin(), order.end(), 0);
        }));
    CHECK(dt3_owner_cells(ascending) == reference);

    dt3_owner descending;
    REQUIRE(
        dt3_build_in_schedule(descending, points, [](tf::buffer<int> &order) {
          std::iota(order.begin(), order.end(), 0);
          std::reverse(order.begin(), order.end());
        }));
    CHECK(dt3_owner_cells(descending) == reference);

    dt3_owner shuffled;
    REQUIRE(
        dt3_build_in_schedule(shuffled, points, [&rng](tf::buffer<int> &order) {
          tf::test::shuffle(order.begin(), order.end(), rng);
        }));
    CHECK(dt3_owner_cells(shuffled) == reference);
  }
}

TEST_CASE("delaunay tets: the order is the levels descending, each in the "
          "sites' Morton storage order",
          "[topology][delaunay3]") {
  namespace dt3 = tf::topology::cdt::dt3;
  dt3_owner owner;
  const auto flat = dt3_random_coordinates<dt3_i32>(2026u, 3000, 100000);
  REQUIRE(dt3::build_tetrahedralization(owner, dt3_lattice_points(flat)));

  const std::size_t n = dt3::tet_site_count(owner);
  REQUIRE(owner._order.size() == n);
  const auto level_of = [&owner](int site) {
    return dt3::tet_site_level(
        dt3::tet_site_priority(std::uint64_t(dt3::tet_site_name(owner, site))));
  };
  // the sites stand in the Morton order of their positions over their domain
  const auto domain =
      dt3::measure_tet_site_domain<false, dt3_i32>(tf::make_mapped_range(
          tf::make_sequence_range(n), [&owner](std::size_t i) {
            return dt3::tet_shallow_point(owner, int(i));
          }));
  const auto shift = dt3::tet_site_morton_shift(domain);
  bool stored_by_key = true;
  for (std::size_t i = 1; i < n; ++i) {
    const auto a = dt3::tet_site_morton_key(
        dt3::tet_shallow_point(owner, int(i - 1)), domain, shift);
    const auto b = dt3::tet_site_morton_key(
        dt3::tet_shallow_point(owner, int(i)), domain, shift);
    stored_by_key =
        stored_by_key &&
        (a < b || (a == b && dt3::tet_site_position(owner, int(i - 1)) <
                                 dt3::tet_site_position(owner, int(i))));
  }
  CHECK(stored_by_key);
  // levels descend along the order and within a level the storage order holds
  std::size_t levels_seen = 0;
  bool descending = true, storage_order = true;
  std::vector<char> present(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const int site = owner._order[i];
    present[std::size_t(site)] = 1;
    if (i == 0) {
      ++levels_seen;
      continue;
    }
    const auto before = level_of(owner._order[i - 1]), now = level_of(site);
    if (now != before) {
      descending = descending && now < before;
      ++levels_seen;
    } else
      storage_order = storage_order && owner._order[i - 1] < site;
  }
  CHECK(levels_seen > 4);
  CHECK(descending);
  CHECK(storage_order);
  CHECK(std::count(present.begin(), present.end(), 1) == std::ptrdiff_t(n));
}

namespace {

/// The default policy's complex at every rung of the worker ladder against
/// the serial policy's: wherever the arena admits the claims, they insert
/// every site the serial prefix left and change nothing the build publishes.
template <typename Int, typename Coord>
auto dt3_check_parallel_coordinates(const std::vector<Coord> &flat) -> void {
  using Tuning = tf::topology::cdt::dt3::tet_execution_tuning;
  REQUIRE(flat.size() / 3 >= Tuning::parallel_sites);
  tf::delaunay_tetrahedralizer<
      int, Coord, Int, tf::topology::cdt::serial_delaunay_execution_policy>
      serial;
  const auto points = dt3_lattice_points(flat);
  REQUIRE(serial.build(points));
  const auto expected = dt3_canonical_cells(serial);
  tf::delaunay_tetrahedralizer<int, Coord, Int> parallel;
  for (const std::size_t workers : tf::test::worker_ladder({1, 2, 4, 8, 16})) {
    CAPTURE(workers, flat.size());
    tbb::task_arena arena(static_cast<int>(workers));
    REQUIRE(arena.execute([&] { return parallel.build(points); }));
    CHECK(parallel.is_valid());
    CHECK(dt3_canonical_cells(parallel) == expected);
    const auto &stats = parallel.stats();
    CHECK((stats.parallel_insertions > 0) ==
          (workers >= Tuning::minimum_workers));
    if (workers >= Tuning::minimum_workers)
      CHECK(stats.insertions - stats.parallel_insertions <=
            Tuning::serial_prefix);
    CHECK(std::equal(serial.index_map().f().begin(),
                     serial.index_map().f().end(),
                     parallel.index_map().f().begin()));
  }
}

} // namespace

TEST_CASE("delaunay tets: parallel claims preserve exact complexes",
          "[.fuzz][topology][delaunay3][parallel]") {
  const std::size_t sites =
      tf::topology::cdt::dt3::tet_execution_tuning::parallel_sites;
  for (std::uint32_t seed = 0; seed < 8; ++seed) {
    CAPTURE(seed);
    dt3_check_parallel_coordinates<dt3_i32>(
        dt3_random_coordinates<dt3_i32>(seed, sites + seed * 17, 200));
  }
  dt3_check_parallel_coordinates<dt3_i64>(
      dt3_random_coordinates<dt3_i64>(42, sites, 1000000000LL));
  auto scaled =
      dt3_random_coordinates<dt3_scaled_coord>(43, sites, 1000000000LL);
  for (auto &coordinate : scaled)
    coordinate *= 64;
  dt3_check_parallel_coordinates<dt3_i32>(scaled);

  std::vector<dt3_i32> grid;
  for (int x = 0; x < 13; ++x)
    for (int y = 0; y < 13; ++y)
      for (int z = 0; z < 13; ++z)
        for (int repeat = 0; repeat < 2; ++repeat)
          grid.insert(grid.end(), {x * 4, y * 4, z * 4});
  dt3_check_parallel_coordinates<dt3_i32>(grid);

  std::vector<dt3_i32> faces;
  for (int i = 0; i < 38; ++i)
    for (int j = 0; j < 38; ++j)
      faces.insert(faces.end(),
                   {i * 8, j * 8, 0, i * 8, 0, j * 8, 0, i * 8, j * 8});
  dt3_check_parallel_coordinates<dt3_i32>(faces);
  if (tf::test::machine_workers() <
      tf::topology::cdt::dt3::tet_execution_tuning::minimum_workers)
    SKIP("the claims need minimum_workers");
}

TEST_CASE("delaunay tets: exhausted pools grow at the barrier",
          "[topology][delaunay3][parallel]") {
  using Tuning = tf::topology::cdt::dt3::tet_execution_tuning;
  using Serial = tf::delaunay_tetrahedralizer<
      int, dt3_i32, dt3_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>;
  // Sites on the moment curve hold far more cells each than the density
  // measured before a band: the streams run dry, and the barriers grow
  // their pools until the bands insert every site past the serial prefix.
  // Repeated, they reach the parallel cutoff and weld.
  std::vector<dt3_i32> curve;
  for (int repeat = 0; repeat < 14; ++repeat)
    for (int t = 0; t < 300; ++t)
      curve.insert(curve.end(), {t, t * t, t * t * t});
  // The same curve inside a cloud: the curve's streams run dry while the
  // cloud's finish and lend them their rows at the barrier.
  auto cloud =
      dt3_random_coordinates<dt3_i32>(273, Tuning::parallel_sites, 10000000);
  for (int t = 0; t < 300; ++t)
    cloud.insert(cloud.end(), {t, t * t, t * t * t});
  for (const bool lent : {false, true}) {
    const auto points = dt3_lattice_points(lent ? cloud : curve);
    Serial serial;
    REQUIRE(serial.build(points));
    const auto expected = dt3_canonical_cells(serial);
    for (const std::size_t workers : tf::test::worker_ladder({1, 4, 8, 16})) {
      CAPTURE(lent, workers);
      tbb::task_arena arena(static_cast<int>(workers));
      dt3_builder<> parallel;
      REQUIRE(arena.execute([&] { return parallel.build(points); }));
      CHECK(parallel.is_valid());
      CHECK(dt3_canonical_cells(parallel) == expected);
      const auto &stats = parallel.stats();
      CHECK(stats.insertions == parallel.n_sites() - 4);
      if (workers < Tuning::minimum_workers) {
        CHECK(stats.parallel_bands == 0);
        continue;
      }
      CHECK(stats.parallel_bands > 0);
      CHECK(stats.pool_restarts > 0);
      CHECK(stats.insertions - stats.parallel_insertions <=
            Tuning::serial_prefix);
    }
  }
  if (tf::test::machine_workers() < Tuning::minimum_workers)
    SKIP("the claims need minimum_workers");
}

TEST_CASE("delaunay tets: public parallel policy reuses and clears its state",
          "[topology][delaunay3][parallel]") {
  using Serial = tf::delaunay_tetrahedralizer<
      int, dt3_i32, dt3_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>;
  const auto flat = dt3_random_coordinates<dt3_i32>(919, 16384, 100000);
  const auto points = dt3_lattice_points(flat);
  Serial serial;
  REQUIRE(serial.build(points));
  const auto expected = dt3_canonical_cells(serial);
  dt3_builder<> parallel;
  for (const std::size_t workers : tf::test::worker_ladder({1, 2, 4, 8, 16})) {
    CAPTURE(workers);
    tbb::task_arena arena(static_cast<int>(workers));
    REQUIRE(arena.execute([&] { return parallel.build(points); }));
    CHECK(parallel.is_valid());
    CHECK(dt3_canonical_cells(parallel) == expected);
    CHECK((parallel.stats().parallel_insertions > 0) ==
          (workers >=
           tf::topology::cdt::dt3::tet_execution_tuning::minimum_workers));
    CHECK(std::equal(serial.index_map().f().begin(),
                     serial.index_map().f().end(),
                     parallel.index_map().f().begin()));
  }
  parallel.clear();
  CHECK(parallel.n_sites() == 0);
  CHECK(parallel.n_tets() == 0);
  CHECK(parallel.stats().parallel_bands == 0);
  const std::vector<dt3_i32> planar{0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4, 0};
  CHECK(!parallel.build(dt3_lattice_points(planar)));
  CHECK(parallel.n_tets() == 0);
  CHECK(parallel.refusal() == tf::tetrahedralization_refusal::rank_deficient);
  const auto small = dt3_random_coordinates<dt3_i32>(71, 20, 4000);
  REQUIRE(parallel.build(dt3_lattice_points(small)));
  CHECK(parallel.is_valid());
  CHECK(parallel.stats().parallel_bands == 0);
}

TEST_CASE("delaunay tets: concurrent capacity refusal publishes no cells",
          "[topology][delaunay3][parallel]") {
  // A 16-bit index names every input slot but not the cells they make, so
  // the claims run dry against the cell space and the tail refuses.
  using Tuning = tf::topology::cdt::dt3::tet_execution_tuning;
  const auto flat =
      dt3_random_coordinates<dt3_i32>(121, Tuning::parallel_sites, 10000);
  const auto points = dt3_lattice_points(flat);
  tf::delaunay_tetrahedralizer<std::int16_t, dt3_i32> narrow;
  for (const std::size_t workers : tf::test::worker_ladder({1, 4, 8, 16})) {
    CAPTURE(workers);
    tbb::task_arena arena(static_cast<int>(workers));
    CHECK(!arena.execute([&] { return narrow.build(points); }));
    CHECK((narrow.stats().parallel_insertions > 0) ==
          (workers >= Tuning::minimum_workers));
    CHECK(narrow.n_tets() == 0);
    CHECK(narrow.refusal() == tf::tetrahedralization_refusal::index_capacity);
    CHECK(!narrow.is_valid());
  }
  if (tf::test::machine_workers() < Tuning::minimum_workers)
    SKIP("the claims need minimum_workers");
}

TEST_CASE("delaunay tets: concurrent rows support wide index tickets",
          "[topology][delaunay3][parallel]") {
  using Tuning = tf::topology::cdt::dt3::tet_execution_tuning;
  const auto flat =
      dt3_random_coordinates<dt3_i32>(37, Tuning::parallel_sites, 10000);
  const auto points = dt3_lattice_points(flat);
  tf::delaunay_tetrahedralizer<
      int, dt3_i32, dt3_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>
      reference;
  REQUIRE(reference.build(points));
  const auto expected = dt3_canonical_cells(reference);
  tf::delaunay_tetrahedralizer<std::int64_t, dt3_i32> wide;
  for (const std::size_t workers : tf::test::worker_ladder({1, 4, 8, 16})) {
    CAPTURE(workers);
    tbb::task_arena arena(static_cast<int>(workers));
    REQUIRE(arena.execute([&] { return wide.build(points); }));
    CHECK((wide.stats().parallel_insertions > 0) ==
          (workers >= Tuning::minimum_workers));
    REQUIRE(wide.is_valid());
    CHECK(dt3_canonical_cells(wide) == expected);
  }
  if (tf::test::machine_workers() < Tuning::minimum_workers)
    SKIP("the claims need minimum_workers");
}

namespace {

auto dt3_rim_with_interior_points(int height) -> std::vector<dt3_i32> {
  std::vector<dt3_i32> points;
  const auto append = [&](double radius, double angle) {
    const auto x = dt3_i32(std::llround(10000 * radius * std::cos(angle)));
    const auto y = dt3_i32(std::llround(10000 * radius * std::sin(angle)));
    const auto z = dt3_i32(std::llround(
        double(height) * (double(x) * x - double(y) * y) / 100000000));
    points.insert(points.end(), {x, y, z});
  };
  const double turn = 2 * std::acos(-1.);
  for (int i = 0; i < 32; ++i)
    append(1, turn * i / 32);
  append(0, 0);
  for (int ring = 0; ring < 3; ++ring) {
    const int count = 256 << ring;
    for (int i = 0; i < count; ++i)
      append((ring + 1) * .3, turn * (i + .25) / count);
  }
  return points;
}


auto dt3_contains_rim_edges(const dt3_owner &owner, int count) -> bool {
  std::vector<std::array<int, 2>> edges;
  for (std::size_t c = 0; c < owner._n_finite; ++c)
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j) {
        int a =
            tf::topology::cdt::dt3::tet_site_name(owner, owner._corners[c][i]);
        int b =
            tf::topology::cdt::dt3::tet_site_name(owner, owner._corners[c][j]);
        if (a < count && b < count)
          edges.push_back({std::min(a, b), std::max(a, b)});
      }
  std::sort(edges.begin(), edges.end());
  for (int i = 0; i < count; ++i) {
    const int j = (i + 1) % count;
    if (!std::binary_search(edges.begin(), edges.end(),
                            std::array<int, 2>{std::min(i, j), std::max(i, j)}))
      return false;
  }
  return true;
}

} // namespace

TEST_CASE("delaunay tets: curved rims accept successive interior point batches",
          "[topology][delaunay3][parallel][rim-interior]") {
  using Tuning = tf::topology::cdt::dt3::tet_execution_tuning;
  using Serial = tf::delaunay_tetrahedralizer<
      int, dt3_i32, dt3_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>;
  // The centre and each ring is a batch, every one past the append's
  // parallel cutoff.
  const std::array<std::size_t, 3> lasts{289, 801, 1825};
  for (int height : {10, 10000}) {
    for (bool distributed : {false, true}) {
      auto flat = dt3_rim_with_interior_points(height);
      REQUIRE(flat.size() / 3 == lasts.back());
      if (distributed) {
        std::vector<std::array<dt3_i32, 3>> interior;
        for (std::size_t i = 32; i < flat.size() / 3; ++i)
          interior.push_back({flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]});
        std::mt19937 random(731);
        tf::test::shuffle(interior.begin(), interior.end(), random);
        for (std::size_t i = 0; i < interior.size(); ++i)
          for (std::size_t k = 0; k < 3; ++k)
            flat[3 * (i + 32) + k] = interior[i][k];
      }
      std::vector<std::vector<std::array<int, 4>>> expected;
      for (const std::size_t last : lasts) {
        Serial reference;
        REQUIRE(reference.build(
            tf::make_points<3>(tf::make_range(flat.data(), last * 3))));
        expected.push_back(dt3_named_cells(reference));
      }
      for (const std::size_t workers :
           tf::test::worker_ladder({1, 2, 4, 8, 16})) {
        CAPTURE(height, distributed, workers);
        tbb::task_arena arena(static_cast<int>(workers));
        dt3_builder<> dt;
        REQUIRE(
            dt.build(tf::make_points<3>(tf::make_range(flat.data(), 32 * 3))));
        REQUIRE(dt3_contains_rim_edges(dt.owner(), 32));
        std::size_t first = 32;
        for (std::size_t b = 0; b < lasts.size(); ++b) {
          const std::size_t last = lasts[b];
          CAPTURE(first, last);
          std::vector<dt3_builder<>::site_type> batch;
          for (auto i = first; i < last; ++i)
            batch.push_back(
                {int(i), {flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]}});
          REQUIRE(batch.size() >= Tuning::parallel_append_sites);
          const auto inserted = dt.stats().parallel_insertions;
          REQUIRE(arena.execute(
              [&] { return dt.append_sites(tf::make_range(batch)); }));
          CHECK((dt.stats().parallel_insertions > inserted) ==
                (workers >= Tuning::minimum_workers));
          REQUIRE(dt.is_valid());
          CHECK(dt3_contains_rim_edges(dt.owner(), 32));
          CHECK(dt3_named_cells(dt) == expected[b]);
          first = last;
        }
      }
    }
  }
}

TEST_CASE("delaunay tets: edge pairing holds on a 64-bit index",
          "[topology][delaunay3]") {
  // The links stamp each site in a dense table sized by the site count,
  // so an index wider than the count changes nothing but the type; the
  // pairing reads equal edges whatever their corners' order, and the bytes
  // an earlier boundary left on the same sites are not read as this one's.
  namespace dt3 = tf::topology::cdt::dt3;
  using Index = std::int64_t;
  dt3::tet_cavity_links<Index> links;
  dt3::open_tet_cavity_links(links, 16);
  dt3::state_tet_cavity_facet(links, {Index(10), Index(9), Index(8)});
  dt3::state_tet_cavity_facet(links, {Index(7), Index(2), Index(1)});
  dt3::open_tet_cavity_links(links, 16);
  dt3::state_tet_cavity_facet(links, {Index(1), Index(2), Index(7)});
  dt3::state_tet_cavity_facet(links, {Index(11), Index(12), Index(8)});
  dt3::state_tet_cavity_facet(links, {Index(2), Index(1), Index(9)});
  dt3::state_tet_cavity_facet(links, {Index(12), Index(11), Index(10)});
  REQUIRE(links.n_facets == 4);
  CHECK(links.across[2] == 8);
  CHECK(links.across[8] == 2);
  CHECK(links.across[5] == 11);
  CHECK(links.across[11] == 5);
  for (std::size_t k : {0, 1, 3, 4, 6, 7, 9, 10})
    CHECK(links.across[k] == -1);
}

namespace {

// A closed boundary of `ring + 2` vertices: two cones over one ring, the
// ring's sites first and the apexes after them.
auto dt3_double_cone(int ring) -> std::vector<std::array<int, 3>> {
  std::vector<std::array<int, 3>> facets;
  for (int i = 0; i < ring; ++i) {
    facets.push_back({ring, i, (i + 1) % ring});
    facets.push_back({ring + 1, (i + 1) % ring, i});
  }
  return facets;
}

// Per facet-edge `3 * i + e`, the facet-edge across the edge opposite
// corner `e`, by brute force over the boundary.
auto dt3_brute_force_across(const std::vector<std::array<int, 3>> &facets)
    -> std::vector<int> {
  const auto edge = [&facets](std::size_t x) {
    const auto &facet = facets[x / 3];
    const int a = facet[(x % 3 + 1) % 3], b = facet[(x % 3 + 2) % 3];
    return std::array<int, 2>{std::min(a, b), std::max(a, b)};
  };
  std::vector<int> across(3 * facets.size(), -1);
  for (std::size_t x = 0; x < across.size(); ++x)
    for (std::size_t y = 0; y < across.size(); ++y)
      if (x != y && edge(x) == edge(y))
        across[x] = int(y);
  return across;
}

} // namespace

TEST_CASE("delaunay tets: edge pairing holds past the local table",
          "[topology][delaunay3]") {
  // A boundary of more vertices than the table names locals for leaves it
  // and pairs from its sorted records; one within it pairs in the table.
  // Successive boundaries over the same links, in either order, pair as if
  // each were the first.
  namespace dt3 = tf::topology::cdt::dt3;
  constexpr int big = int(dt3::tet_cavity_links<int>::max_local) + 44;
  for (const auto &rings :
       {std::array<int, 3>{100, big, 100}, std::array<int, 3>{big, 100, big}}) {
    dt3::tet_cavity_links<int> links;
    for (const int ring : rings) {
      CAPTURE(ring);
      const auto facets = dt3_double_cone(ring);
      dt3::open_tet_cavity_links(links, std::size_t(big + 2));
      for (const auto &facet : facets)
        dt3::state_tet_cavity_facet(links, facet);
      CHECK(links.overflowed ==
            (std::size_t(ring + 2) > dt3::tet_cavity_links<int>::max_local));
      dt3::close_tet_cavity_links(links);
      REQUIRE(links.n_facets == facets.size());
      const auto expected = dt3_brute_force_across(facets);
      CHECK(std::equal(expected.begin(), expected.end(), links.across.begin()));
    }
  }
}

TEST_CASE("delaunay tets: input slots refuse before narrowing",
          "[topology][delaunay3][capacity]") {
  tf::delaunay_tetrahedralizer<std::int8_t, dt3_i32> tiny;
  const auto flat = dt3_random_coordinates<dt3_i32>(913, 200, 10000);
  REQUIRE_FALSE(tiny.build(dt3_lattice_points(flat)));
  CHECK(tiny.refusal() == tf::tetrahedralization_refusal::index_capacity);
  CHECK(tiny.n_tets() == 0);
  CHECK(tiny.n_sites() == 0);
  const std::vector<dt3_i32> duplicates(600, 0);
  REQUIRE_FALSE(tiny.build(dt3_lattice_points(duplicates)));
  CHECK(tiny.refusal() == tf::tetrahedralization_refusal::index_capacity);
  CHECK(tiny.n_tets() == 0);
}

TEST_CASE("delaunay tets compose with tetrahedra geometry views",
          "[topology][delaunay3]") {
  const auto flat = dt3_random_coordinates<dt3_i32>(9123, 48, 1000);
  dt3_builder<> dt;
  REQUIRE(dt.build(dt3_lattice_points(flat)));
  STATIC_REQUIRE(std::is_same_v<decltype(dt.tets()),
                                decltype(tf::make_tetras(dt.tets()))>);
  const auto tetras = dt.tets();
  const auto points = tf::make_mapped_range(
      dt.sites(), [](const auto &site) -> const auto & { return site.pt; });
  const auto tetrahedra = tf::make_tetrahedra(tetras, points);
  const auto buffer = tf::make_tetrahedra_buffer(tetrahedra);
  REQUIRE(buffer.size() == dt.n_tets());
  REQUIRE(buffer.points().size() == dt.n_sites());
  for (std::size_t i = 0; i < buffer.size(); ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      REQUIRE(buffer.tetras()[i][j] == tetras[i][j]);
      for (std::size_t axis = 0; axis < 3; ++axis)
        REQUIRE(buffer.tetrahedra()[i][j][axis] ==
                dt.sites()[tetras[i][j]].pt[axis]);
    }
  }
}

TEST_CASE("delaunay tets: admitted append preserves slots and input aliases",
          "[topology][delaunay3][append][parallel]") {
  for (const std::size_t workers : tf::test::worker_ladder({1, 4, 16})) {
    CAPTURE(workers);
    dt3_builder<> dt;
    std::vector<dt3_builder<>::site_type> sites;
    for (int i = 0; i < 625; ++i)
      sites.push_back(
          {1000 + 7 * i, {(i % 5) * 64, ((i / 5) % 5) * 64, (i / 25) * 64}});
    std::vector<dt3_builder<>::site_type> initial(sites.begin(),
                                                  sites.begin() + 32);
    initial.push_back(initial[3]);
    REQUIRE(dt.build(tf::make_range(initial)));
    std::vector<int> original_map(dt.index_map().f().begin(),
                                  dt.index_map().f().end());
    std::vector<dt3_builder<>::site_type> old_sites(dt.sites().begin(),
                                                    dt.sites().end());
    tbb::task_arena arena(static_cast<int>(workers));
    std::size_t first = 32;
    for (std::size_t last : {std::size_t(288), std::size_t(625)}) {
      REQUIRE(arena.execute([&] {
        return dt.append_sites(
            tf::make_range(sites.data() + first, last - first));
      }));
      REQUIRE(dt.is_valid());
      REQUIRE(dt.n_sites() == last);
      REQUIRE(dt.index_map().f().size() == last + 1);
      for (std::size_t i = 0; i < original_map.size(); ++i)
        CHECK(dt.index_map().f()[i] == original_map[i]);
      for (std::size_t i = 0; i < old_sites.size(); ++i) {
        CHECK(dt.sites()[i].id == old_sites[i].id);
        CHECK(dt.sites()[i].pt == old_sites[i].pt);
      }
      for (std::size_t i = first; i < last; ++i) {
        CHECK(dt.index_map().f()[i + 1] == int(i));
        CHECK(dt.index_map().kept_ids()[i] == int(i + 1));
      }
      dt3_builder<> reference;
      REQUIRE(reference.build(tf::make_range(sites.data(), last)));
      CHECK(dt3_named_cells(dt) == dt3_named_cells(reference));
      if (last == 288)
        CHECK(dt3_spheres_are_empty<dt3_i32>(dt));
      first = last;
    }
    if (workers >=
        tf::topology::cdt::dt3::tet_execution_tuning::minimum_workers)
      CHECK(dt.stats().parallel_insertions > 0);
    const auto before = dt3_named_cells(dt);
    REQUIRE(dt.append_sites(tf::make_range(sites.data(), std::size_t(0))));
    CHECK(dt3_named_cells(dt) == before);
  }
}

TEST_CASE("delaunay tets: append capacity refusal publishes no cells",
          "[topology][delaunay3][append]") {
  using DT = tf::delaunay_tetrahedralizer<std::int8_t, dt3_i32>;
  std::vector<DT::site_type> sites{
      {0, {0, 0, 0}}, {1, {64, 0, 0}}, {2, {0, 64, 0}}, {3, {0, 0, 64}}};
  DT dt;
  REQUIRE(dt.build(tf::make_range(sites)));
  std::vector<DT::site_type> extra;
  for (int i = 4; i < 132; ++i)
    extra.push_back({std::int8_t(i % 127), {i, i * i, i * i * i}});
  REQUIRE_FALSE(dt.append_sites(tf::make_range(extra)));
  CHECK(dt.refusal() == tf::tetrahedralization_refusal::index_capacity);
  CHECK(dt.n_tets() == 0);
}

TEST_CASE("delaunay tets: serial policy appends without claim workspace",
          "[topology][delaunay3][append]") {
  using DT = tf::delaunay_tetrahedralizer<
      int, dt3_i32, dt3_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>;
  std::vector<DT::site_type> sites{{0, {0, 0, 0}},
                                   {1, {64, 0, 0}},
                                   {2, {0, 64, 0}},
                                   {3, {0, 0, 64}},
                                   {4, {8, 8, 8}}};
  DT dt;
  REQUIRE(dt.build(tf::make_range(sites.data(), std::size_t(4))));
  REQUIRE(dt.append_sites(tf::make_range(sites.data() + 4, std::size_t(1))));
  CHECK(dt.is_valid());
  CHECK(dt.stats().parallel_insertions == 0);
  CHECK(dt3_spheres_are_empty<dt3_i32>(dt));
}

TEST_CASE(
    "delaunay tets: parallel dispatch preserves the complex at its cutoff",
    "[topology][delaunay3][parallel]") {
  using Serial = tf::delaunay_tetrahedralizer<
      int, dt3_i32, dt3_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>;
  using Tuning = tf::topology::cdt::dt3::tet_execution_tuning;
  for (std::size_t count : {Tuning::parallel_sites - 1, Tuning::parallel_sites,
                            Tuning::parallel_sites + 1}) {
    CAPTURE(count);
    const auto flat = dt3_random_coordinates<dt3_i32>(731, count, 100000);
    const auto points = dt3_lattice_points(flat);
    Serial serial;
    REQUIRE(serial.build(points));
    const auto expected = dt3_canonical_cells(serial);
    for (const std::size_t workers : tf::test::worker_ladder({1, 4, 16})) {
      CAPTURE(workers);
      tbb::task_arena arena(static_cast<int>(workers));
      dt3_builder<> parallel;
      REQUIRE(arena.execute([&] { return parallel.build(points); }));
      CHECK(parallel.is_valid());
      CHECK(dt3_canonical_cells(parallel) == expected);
      CHECK((parallel.stats().parallel_insertions > 0) ==
            (count >= Tuning::parallel_sites &&
             workers >= Tuning::minimum_workers));
      CHECK(std::equal(serial.index_map().f().begin(),
                       serial.index_map().f().end(),
                       parallel.index_map().f().begin()));
    }
  }
  if (tf::test::machine_workers() < Tuning::minimum_workers)
    SKIP("the claims need minimum_workers");
}

namespace {

/// The published cells, then their links, as one flat stream in the order
/// and the slot order they are published in.
template <typename Builder>
auto dt3_published_stream(const Builder &dt) -> std::vector<int> {
  std::vector<int> stream;
  for (auto cell : dt.tets())
    for (std::size_t slot = 0; slot < 4; ++slot)
      stream.push_back(int(cell[slot]));
  for (auto links : dt.neighbors())
    for (std::size_t slot = 0; slot < 4; ++slot)
      stream.push_back(int(links[slot]));
  return stream;
}

} // namespace

TEST_CASE("delaunay tets: the published stream is one at any worker count",
          "[.fuzz][topology][delaunay3][parallel][determinism]") {
  using Serial = tf::delaunay_tetrahedralizer<
      int, dt3_i32, dt3_i32,
      tf::topology::cdt::serial_delaunay_execution_policy>;
  // The cospherical grid is where a tie-break could follow insertion timing.
  std::vector<dt3_i32> grid;
  for (int x = 0; x < 17; ++x)
    for (int y = 0; y < 17; ++y)
      for (int z = 0; z < 17; ++z)
        grid.insert(grid.end(), {x * 4, y * 4, z * 4});
  const std::vector<std::vector<dt3_i32>> sources{
      dt3_random_coordinates<dt3_i32>(5150u, 16384, 1000000), grid};
  for (const auto &flat : sources) {
    const auto points = dt3_lattice_points(flat);
    Serial serial;
    REQUIRE(serial.build(points));
    const auto expected = dt3_published_stream(serial);
    for (const std::size_t workers : tf::test::worker_ladder({1, 2, 4, 16})) {
      CAPTURE(flat.size(), workers);
      tbb::task_arena arena(static_cast<int>(workers));
      dt3_builder<> parallel;
      REQUIRE(arena.execute([&] { return parallel.build(points); }));
      CHECK((parallel.stats().parallel_insertions > 0) ==
            (workers >=
             tf::topology::cdt::dt3::tet_execution_tuning::minimum_workers));
      CHECK(dt3_published_stream(parallel) == expected);
    }
  }

  using Site = dt3_builder<>::site_type;
  const auto flat = dt3_random_coordinates<dt3_i32>(5151u, 16384, 1000000);
  std::vector<Site> sites;
  for (std::size_t i = 0; i < flat.size() / 3; ++i)
    sites.push_back({int(i), {flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]}});
  const auto initial = tf::make_range(sites.data(), std::size_t(12288));
  const auto appended = tf::make_range(sites.data() + 12288, std::size_t(4096));
  Serial serial;
  REQUIRE(serial.build(initial));
  REQUIRE(serial.append_sites(appended));
  const auto expected = dt3_published_stream(serial);
  for (const std::size_t workers : tf::test::worker_ladder({1, 2, 4, 16})) {
    CAPTURE(workers);
    tbb::task_arena arena(static_cast<int>(workers));
    dt3_builder<> parallel;
    REQUIRE(arena.execute([&] {
      return parallel.build(initial) && parallel.append_sites(appended);
    }));
    CHECK((parallel.stats().parallel_insertions > 0) ==
          (workers >=
           tf::topology::cdt::dt3::tet_execution_tuning::minimum_workers));
    CHECK(dt3_published_stream(parallel) == expected);
  }
  if (tf::test::machine_workers() <
      tf::topology::cdt::dt3::tet_execution_tuning::minimum_workers)
    SKIP("the claims need minimum_workers");
}

namespace {

using dt3_site = dt3_owner::site_type;

/// The finite cells' facets, each three slots ascending, the set ascending.
auto dt3_barrier_facets(const dt3_owner &owner)
    -> std::vector<std::array<int, 3>> {
  std::vector<std::array<int, 3>> facets;
  for (std::size_t c = 0; c < owner._n_finite; ++c)
    for (std::size_t slot = 0; slot < 4; ++slot) {
      auto facet = tf::topology::cdt::dt3::tet_facet(owner._corners[c], slot);
      std::sort(facet.begin(), facet.end());
      facets.push_back(facet);
    }
  std::sort(facets.begin(), facets.end());
  facets.erase(std::unique(facets.begin(), facets.end()), facets.end());
  return facets;
}

auto dt3_barrier_holds(const dt3_owner &owner, const std::array<int, 3> &facet)
    -> bool {
  const auto facets = dt3_barrier_facets(owner);
  return std::binary_search(facets.begin(), facets.end(), facet);
}

/// The owner's published cells, then their links, in published order.
auto dt3_barrier_stream(const dt3_owner &owner) -> std::vector<int> {
  std::vector<int> stream;
  for (std::size_t c = 0; c < owner._n_finite; ++c)
    for (std::size_t slot = 0; slot < 4; ++slot)
      stream.push_back(owner._corners[c][slot]);
  for (std::size_t c = 0; c < owner._n_finite; ++c)
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const int link = owner._neighbors[c][slot];
      stream.push_back(std::size_t(link) < owner._n_finite ? link : -1);
    }
  return stream;
}

auto dt3_barrier_build(dt3_owner &owner, const std::vector<dt3_i32> &flat)
    -> void {
  REQUIRE(tf::topology::cdt::dt3::build_tetrahedralization(
      owner, dt3_lattice_points(flat)));
}

auto dt3_barrier_conflicts(const std::array<dt3_site, 4> &cell,
                           const dt3_site &site) -> bool {
  const std::array<dt3_site, 5> vs{cell[0], cell[1], cell[2], cell[3], site};
  return tf::exact::insphere_conflict_scaled<dt3_i32>(vs.data(), 1);
}

/// A facet two finite cells share and a site on the near cell's side that
/// both cells' circumspheres hold while no other neighbour of the far cell's
/// does: the unbarred cavity swallows the facet, and the far cell is entered
/// across the facet alone. `guards` are the far cell and its neighbours; a
/// site none of them holds never reaches the far cell. The build is
/// deterministic, so every owner rebuilt from `flat` names the same cells.
struct dt3_barrier_case {
  std::vector<dt3_i32> flat;
  int far_cell = -1;
  int near_cell = -1;
  std::array<int, 3> facet{};
  std::array<int, 4> far_corners{};
  std::vector<std::array<dt3_site, 4>> guards;
  dt3_site site{};
};

auto dt3_barrier_find_case() -> dt3_barrier_case {
  namespace dt3 = tf::topology::cdt::dt3;
  dt3_barrier_case found;
  for (std::uint32_t seed = 4021; seed < 4085; ++seed) {
    found.flat = dt3_random_coordinates<dt3_i32>(seed, 40, 1 << 20);
    dt3_owner owner;
    dt3_barrier_build(owner, found.flat);
    const auto cell_sites = [&owner](int cell) {
      const auto corners = owner._corners[std::size_t(cell)];
      return std::array<dt3_site, 4>{dt3::tet_shallow_site(owner, corners[0]),
                                     dt3::tet_shallow_site(owner, corners[1]),
                                     dt3::tet_shallow_site(owner, corners[2]),
                                     dt3::tet_shallow_site(owner, corners[3])};
    };
    for (std::size_t c = 0; c < owner._n_finite; ++c)
      for (std::size_t slot = 0; slot < 4; ++slot) {
        const int far_cell = owner._neighbors[c][slot];
        if (std::size_t(far_cell) >= owner._n_finite)
          continue;
        std::vector<std::array<dt3_site, 4>> guards{cell_sites(far_cell)};
        bool finite = true;
        for (std::size_t k = 0; k < 4; ++k) {
          const int beside = owner._neighbors[std::size_t(far_cell)][k];
          finite = finite && std::size_t(beside) < owner._n_finite;
          if (beside != int(c))
            guards.push_back(cell_sites(beside));
        }
        if (!finite)
          continue;
        const auto near_sites = cell_sites(int(c));
        const auto facet = dt3::tet_facet(owner._corners[c], slot);
        const auto &apex = near_sites[slot];
        std::array<dt3_site, 3> wall{};
        for (std::size_t k = 0; k < 3; ++k)
          wall[k] = dt3::tet_shallow_site(owner, facet[k]);
        for (long long step = 1; step < 64; step *= 2) {
          dt3_site site{int(owner._next_name), {}};
          for (std::size_t k = 0; k < 3; ++k) {
            const long long sum =
                (long long)wall[0].pt[k] + wall[1].pt[k] + wall[2].pt[k];
            site.pt[k] =
                dt3_i32(((64 - step) * sum + 3 * step * apex.pt[k]) / 192);
          }
          const auto side = [&](const tf::exact::pt3<dt3_i32> &p) {
            return tf::exact::orient3d_sign<dt3_i32>(wall[0].pt, wall[1].pt,
                                                     wall[2].pt, p);
          };
          if (side(site.pt) != side(apex.pt) ||
              !dt3_barrier_conflicts(near_sites, site) ||
              !dt3_barrier_conflicts(guards[0], site))
            continue;
          bool entered_elsewhere = false;
          for (std::size_t g = 1; g < guards.size(); ++g)
            entered_elsewhere =
                entered_elsewhere || dt3_barrier_conflicts(guards[g], site);
          if (entered_elsewhere)
            continue;
          found.far_cell = far_cell;
          found.near_cell = int(c);
          found.facet = facet;
          std::sort(found.facet.begin(), found.facet.end());
          const auto far_row = owner._corners[std::size_t(far_cell)];
          found.far_corners = {far_row[0], far_row[1], far_row[2], far_row[3]};
          std::sort(found.far_corners.begin(), found.far_corners.end());
          found.guards = guards;
          found.site = site;
          return found;
        }
      }
  }
  return found;
}

} // namespace

TEST_CASE("delaunay tets: a protected facet bars the cavity a plain append "
          "floods",
          "[topology][delaunay3][barrier]") {
  namespace dt3 = tf::topology::cdt::dt3;
  const auto found = dt3_barrier_find_case();
  REQUIRE(found.far_cell >= 0);
  const std::vector<dt3_site> sites{found.site};
  tf::none_t serial{};

  dt3_owner stock;
  dt3_barrier_build(stock, found.flat);
  REQUIRE(dt3_barrier_holds(stock, found.facet));
  REQUIRE(dt3::append_tetrahedralization_sites<false>(
      stock, tf::make_range(sites), serial));
  dt3::compact_tetrahedralization<false, true>(stock);
  REQUIRE(dt3::check_tetrahedralization_validity(stock));
  CHECK_FALSE(dt3_barrier_holds(stock, found.facet));

  dt3_owner barred;
  dt3_barrier_build(barred, found.flat);
  const int slot = int(tf::topology::cdt::dt3::tet_site_count(barred));
  dt3::protect_tet_facets(barred, std::vector<std::array<int, 3>>{found.facet});
  REQUIRE(dt3::append_tetrahedralization_sites<false>(
      barred, tf::make_range(sites), serial));
  dt3::compact_tetrahedralization<false, true>(barred);
  REQUIRE(dt3::check_tetrahedralization_validity(barred));
  CHECK(dt3_barrier_holds(barred, found.facet));
  const auto cells = dt3_owner_cells(barred);
  CHECK(std::binary_search(cells.begin(), cells.end(), found.far_corners));
  auto star =
      std::array<int, 4>{found.facet[0], found.facet[1], found.facet[2], slot};
  std::sort(star.begin(), star.end());
  CHECK(std::binary_search(cells.begin(), cells.end(), star));
}

TEST_CASE("delaunay tets: a protected facet the cavity never reaches changes "
          "nothing",
          "[topology][delaunay3][barrier]") {
  namespace dt3 = tf::topology::cdt::dt3;
  const auto found = dt3_barrier_find_case();
  REQUIRE(found.far_cell >= 0);
  const std::vector<dt3_site> sites{found.site};
  tf::none_t serial{};

  dt3_owner stock;
  dt3_barrier_build(stock, found.flat);
  const auto facets = dt3_barrier_facets(stock);
  std::array<int, 3> absent{0, 1, 2};
  while (std::binary_search(facets.begin(), facets.end(), absent))
    ++absent[2];
  REQUIRE(dt3::append_tetrahedralization_sites<false>(
      stock, tf::make_range(sites), serial));
  dt3::compact_tetrahedralization<false, true>(stock);

  dt3_owner barred;
  dt3_barrier_build(barred, found.flat);
  dt3::protect_tet_facets(barred, std::vector<std::array<int, 3>>{absent});
  REQUIRE(dt3::tet_protected_count(barred) == 1);
  REQUIRE(dt3::append_tetrahedralization_sites<false>(
      barred, tf::make_range(sites), serial));
  dt3::compact_tetrahedralization<false, true>(barred);
  CHECK(dt3_barrier_stream(barred) == dt3_barrier_stream(stock));
}

TEST_CASE("delaunay tets: a protected lane appends serially and keeps its "
          "facet",
          "[topology][delaunay3][barrier][parallel]") {
  namespace dt3 = tf::topology::cdt::dt3;
  const auto found = dt3_barrier_find_case();
  REQUIRE(found.far_cell >= 0);
  const auto extra = dt3_random_coordinates<dt3_i32>(4022, 1200, 1 << 20);
  std::vector<dt3_site> sites{found.site};
  for (std::size_t i = 0; i < extra.size() / 3; ++i) {
    const dt3_site site{found.site.id + 1 + int(i),
                        {extra[3 * i], extra[3 * i + 1], extra[3 * i + 2]}};
    bool reaches = false;
    for (const auto &guard : found.guards)
      reaches = reaches || dt3_barrier_conflicts(guard, site);
    if (!reaches)
      sites.push_back(site);
  }
  REQUIRE(sites.size() >= dt3::tet_execution_tuning::parallel_append_sites);
  std::vector<int> expected;
  for (const std::size_t workers : tf::test::worker_ladder({1, 4, 8})) {
    CAPTURE(workers);
    dt3_owner owner;
    dt3_barrier_build(owner, found.flat);
    dt3::protect_tet_facets(owner,
                            std::vector<std::array<int, 3>>{found.facet});
    dt3::tet_claim_workspace<int> work{};
    tbb::task_arena arena(static_cast<int>(workers));
    REQUIRE(arena.execute([&] {
      return dt3::append_tetrahedralization_sites<true>(
          owner, tf::make_range(sites), work);
    }));
    CHECK(owner._stats.parallel_insertions == 0);
    dt3::compact_tetrahedralization<false, true>(owner);
    REQUIRE(dt3::check_tetrahedralization_validity(owner));
    CHECK(dt3_barrier_holds(owner, found.facet));
    if (expected.empty())
      expected = dt3_barrier_stream(owner);
    CHECK(dt3_barrier_stream(owner) == expected);
  }
}

namespace {

/// A facet two finite cells share and a site just off its centroid inside
/// the near cell, whose cavity the barred flood reaches around the facet's
/// edges: the facet is wrapped. `flat` rebuilds the same cells.
struct dt3_wrap_case {
  std::vector<dt3_i32> flat;
  std::array<int, 3> facet{};
  dt3_site site{};
};

auto dt3_wrap_find_case() -> dt3_wrap_case {
  namespace dt3 = tf::topology::cdt::dt3;
  dt3_wrap_case found;
  for (std::uint32_t seed = 4021; seed < 4085; ++seed) {
    found.flat = dt3_random_coordinates<dt3_i32>(seed, 40, 1 << 20);
    dt3_owner owner;
    dt3_barrier_build(owner, found.flat);
    for (std::size_t c = 0; c < owner._n_finite; ++c)
      for (std::size_t slot = 0; slot < 4; ++slot) {
        if (std::size_t(owner._neighbors[c][slot]) >= owner._n_finite)
          continue;
        auto facet = dt3::tet_facet(owner._corners[c], slot);
        const auto &apex =
            dt3::tet_shallow_site(owner, owner._corners[c][slot]);
        dt3_site site{int(owner._next_name), {}};
        for (std::size_t k = 0; k < 3; ++k) {
          const long long sum =
              (long long)dt3::tet_shallow_point(owner, facet[0])[k] +
              dt3::tet_shallow_point(owner, facet[1])[k] +
              dt3::tet_shallow_point(owner, facet[2])[k];
          site.pt[k] = dt3_i32((63 * sum + 3 * (long long)apex.pt[k]) / 192);
        }
        std::sort(facet.begin(), facet.end());
        dt3_owner trial;
        dt3_barrier_build(trial, found.flat);
        dt3::protect_tet_facets(trial, std::vector<std::array<int, 3>>{facet});
        tf::none_t serial{};
        const std::vector<dt3_site> sites{site};
        const auto input = trial._index_map.f().size();
        if (!dt3::append_tetrahedralization_sites<false>(
                trial, tf::make_range(sites), serial) ||
            trial._index_map.f()[input] != dt3_owner::none)
          continue;
        found.facet = facet;
        found.site = site;
        return found;
      }
  }
  found.flat.clear();
  return found;
}

} // namespace

TEST_CASE("delaunay tets: a site whose cavity wraps a protected facet is "
          "withdrawn",
          "[topology][delaunay3][barrier][wrap]") {
  namespace dt3 = tf::topology::cdt::dt3;
  const auto found = dt3_wrap_find_case();
  REQUIRE(!found.flat.empty());
  const std::vector<dt3_site> sites{found.site};
  tf::none_t serial{};

  dt3_owner stock;
  dt3_barrier_build(stock, found.flat);
  REQUIRE(dt3::append_tetrahedralization_sites<false>(
      stock, tf::make_range(sites), serial));
  dt3::compact_tetrahedralization<false, true>(stock);
  CHECK_FALSE(dt3_barrier_holds(stock, found.facet));

  dt3_owner barred;
  dt3_barrier_build(barred, found.flat);
  dt3::protect_tet_facets(barred, std::vector<std::array<int, 3>>{found.facet});
  const auto before = dt3_barrier_stream(barred);
  const auto n_sites = dt3::tet_site_count(barred);
  const auto input = barred._index_map.f().size();
  const auto insertions = barred._stats.insertions;
  REQUIRE(dt3::append_tetrahedralization_sites<false>(
      barred, tf::make_range(sites), serial));
  dt3::compact_tetrahedralization<false, true>(barred);
  REQUIRE(dt3::check_tetrahedralization_validity(barred));
  CHECK(barred._index_map.f()[input] == dt3_owner::none);
  CHECK(dt3::tet_site_count(barred) == n_sites);
  CHECK(barred._index_map.kept_ids().size() == n_sites);
  CHECK(barred._stats.insertions == insertions);
  CHECK(dt3_barrier_stream(barred) == before);
  CHECK(dt3_barrier_holds(barred, found.facet));
}

TEST_CASE("delaunay tets: a withdrawn site closes up the slots after it",
          "[topology][delaunay3][barrier][wrap]") {
  namespace dt3 = tf::topology::cdt::dt3;
  const auto found = dt3_wrap_find_case();
  REQUIRE(!found.flat.empty());
  tf::none_t serial{};
  std::vector<dt3_site> others;
  for (int i = 0; i < 6; ++i)
    others.push_back({found.site.id + 1 + i,
                      {(1 << 22) + 4096 * i, 8192 * (i % 2), 4096 * (i / 3)}});
  std::vector<dt3_site> batch{others[0], others[1], found.site};
  batch.insert(batch.end(), others.begin() + 2, others.end());

  dt3_owner expected;
  dt3_barrier_build(expected, found.flat);
  dt3::protect_tet_facets(expected,
                          std::vector<std::array<int, 3>>{found.facet});
  REQUIRE(dt3::append_tetrahedralization_sites<false>(
      expected, tf::make_range(others), serial));
  dt3::compact_tetrahedralization<false, true>(expected);

  dt3_owner owner;
  dt3_barrier_build(owner, found.flat);
  dt3::protect_tet_facets(owner, std::vector<std::array<int, 3>>{found.facet});
  const auto first = dt3::tet_site_count(owner);
  const auto input = owner._index_map.f().size();
  REQUIRE(dt3::append_tetrahedralization_sites<false>(
      owner, tf::make_range(batch), serial));
  dt3::compact_tetrahedralization<false, true>(owner);
  REQUIRE(dt3::check_tetrahedralization_validity(owner));
  CHECK(dt3::tet_site_count(owner) == first + others.size());
  CHECK(owner._index_map.f()[input + 2] == dt3_owner::none);
  for (std::size_t i = 0; i < batch.size(); ++i) {
    if (i == 2)
      continue;
    const auto slot = std::size_t(owner._index_map.f()[input + i]);
    CHECK(slot == first + (i < 2 ? i : i - 1));
    CHECK(dt3::tet_site_name(owner, int(slot)) == batch[i].id);
    CHECK(std::size_t(owner._index_map.kept_ids()[slot]) == input + i);
  }
  CHECK(dt3_owner_cells(owner) == dt3_owner_cells(expected));
  CHECK(dt3_barrier_holds(owner, found.facet));
}
