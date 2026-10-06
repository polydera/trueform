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
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/tetrahedra_buffer.hpp>
#include <trueform/core/tetras.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/exact/insphere.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/orient3d.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/cdt/dt3/assign_tet_compaction.hpp>
#include <trueform/topology/cdt/dt3/bootstrap_tet_claims.hpp>
#include <trueform/topology/cdt/dt3/build_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/check_tetrahedralization_validity.hpp>
#include <trueform/topology/cdt/dt3/clear_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/clear_tetrahedralization_cells.hpp>
#include <trueform/topology/cdt/dt3/compact_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/encode_tet_site_keys.hpp>
#include <trueform/topology/cdt/dt3/finish_tet_claims.hpp>
#include <trueform/topology/cdt/dt3/insert_tet_claim_band.hpp>
#include <trueform/topology/cdt/dt3/insert_tetrahedralization_site.hpp>
#include <trueform/topology/cdt/dt3/load_tetrahedralization_sites.hpp>
#include <trueform/topology/cdt/dt3/measure_tet_site_domain.hpp>
#include <trueform/topology/cdt/dt3/order_tetrahedralization_sites.hpp>
#include <trueform/topology/cdt/dt3/prepare_tet_claim_pool.hpp>
#include <trueform/topology/cdt/dt3/prepare_tet_claim_sites.hpp>
#include <trueform/topology/cdt/dt3/prepare_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/seed_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/tet_cavity_edge_table.hpp>
#include <trueform/topology/cdt/dt3/tet_claim_scratch.hpp>
#include <trueform/topology/cdt/dt3/tet_claim_workspace.hpp>
#include <trueform/topology/cdt/dt3/tet_conflicts.hpp>
#include <trueform/topology/cdt/dt3/tet_execution_tuning.hpp>
#include <trueform/topology/cdt/dt3/tet_facets.hpp>
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
#include <trueform/topology/cdt/dt3/find_tet_site_coincidence.hpp>
#include <trueform/topology/cdt/dt3/propose_tet_edge_midpoints.hpp>
#include <trueform/topology/cdt/dt3/state_missing_tet_edges.hpp>
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
  std::uniform_int_distribution<long long> axis(-(long long)(span),
                                                (long long)(span));
  std::vector<Coord> flat;
  flat.reserve(sites * 3);
  for (std::size_t i = 0; i < sites * 3; ++i)
    flat.push_back(Coord(axis(rng)));
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
  dt3::tet_cavity_edge_table<int> edges{};
  for (std::size_t i = 0; i < owner._order.size(); ++i) {
    const int site = owner._order[i];
    if (site == seeded[0] || site == seeded[1] || site == seeded[2] ||
        site == seeded[3])
      continue;
    if (!dt3::insert_tetrahedralization_site(owner, site, edges))
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
          "[topology][delaunay3]") {
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
  // One compaction always closes a build, so more than one says the dead
  // slots were swept mid-build too.
  CHECK(first.stats().compactions > 1);
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
  // of itself, so the complex is the unscaled one.
  CHECK(dt3_canonical_cells(by_scale) == dt3_canonical_cells(by_points));
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
          std::shuffle(order.begin(), order.end(), rng);
        }));
    CHECK(dt3_owner_cells(shuffled) == reference);
  }
}

TEST_CASE("delaunay tets: the bands are the ranking's and the code orders "
          "inside them",
          "[topology][delaunay3]") {
  dt3_owner owner;
  const auto flat = dt3_random_coordinates<dt3_i32>(2026u, 3000, 100000);
  REQUIRE(tf::topology::cdt::dt3::build_tetrahedralization(
      owner, dt3_lattice_points(flat)));

  const std::size_t n = owner._sites.size();
  std::vector<int> ranked(n);
  std::iota(ranked.begin(), ranked.end(), 0);
  std::sort(ranked.begin(), ranked.end(), [&owner](int a, int b) {
    const auto x = std::size_t(a), y = std::size_t(b);
    const auto key_x = tf::topology::cdt::dt3::tet_site_priority(
        std::uint64_t(owner._sites[x].id));
    const auto key_y = tf::topology::cdt::dt3::tet_site_priority(
        std::uint64_t(owner._sites[y].id));
    return key_x < key_y ||
           (key_x == key_y && owner._sites[x].id < owner._sites[y].id);
  });

  std::size_t bands = 0;
  bool membership_holds = true, codes_ascend = true;
  for (std::size_t last = n; last != 0; last >>= 1) {
    const std::size_t first = last >> 1;
    std::vector<int> expected(ranked.begin() + std::ptrdiff_t(first),
                              ranked.begin() + std::ptrdiff_t(last));
    std::vector<int> stated(owner._order.begin() + std::ptrdiff_t(first),
                            owner._order.begin() + std::ptrdiff_t(last));
    std::sort(expected.begin(), expected.end());
    std::sort(stated.begin(), stated.end());
    membership_holds = membership_holds && stated == expected;
    for (std::size_t i = first + 1; i < last; ++i)
      codes_ascend =
          codes_ascend && owner._keys[std::size_t(owner._order[i - 1])] <=
                              owner._keys[std::size_t(owner._order[i])];
    ++bands;
  }
  CHECK(bands > 8);
  CHECK(membership_holds);
  CHECK(codes_ascend);
}

namespace {

template <typename Owner> struct dt3_parallel_view {
  using site_type = typename Owner::site_type;
  const Owner &owner;
  auto sites() const { return tf::make_range(owner._sites); }
  auto tets() const {
    return tf::make_range(owner._corners.begin(),
                          owner._corners.begin() + owner._n_finite);
  }
};

template <typename Owner, typename Points>
auto dt3_forced_parallel_build(Owner &owner, const Points &points,
                               std::size_t reserve_per_site = 16) -> bool {
  namespace dt3 = tf::topology::cdt::dt3;
  using Index = typename Owner::index_type;
  if (!dt3::prepare_tet_claim_sites(owner, points))
    return false;
  REQUIRE(dt3::bootstrap_tet_claims(owner, 0));
  dt3::tet_claim_workspace<Index> work{};
  const auto live = owner._corners.size();
  const auto available = std::size_t(Owner::max_tets) - live;
  const auto extra =
      reserve_per_site && owner._sites.size() > available / reserve_per_site
          ? available
          : owner._sites.size() * reserve_per_site;
  dt3::prepare_tet_claim_pool(owner, work, 8, live + extra);
  dt3::insert_tet_claim_band(owner, work, 0, owner._order.size());
  return dt3::finish_tet_claims(owner, work);
}

template <typename Int, typename Coord>
auto dt3_check_parallel_coordinates(const std::vector<Coord> &flat) -> void {
  using Owner =
      tf::topology::cdt::dt3::tetrahedralization_owner<int, Coord, Int>;
  tf::delaunay_tetrahedralizer<
      int, Coord, Int, tf::topology::cdt::serial_delaunay_execution_policy>
      serial;
  const auto points = dt3_lattice_points(flat);
  REQUIRE(serial.build(points));
  const auto expected = dt3_canonical_cells(serial);
  Owner owner;
  for (int workers : {1, 2, 4, 8, 16}) {
    CAPTURE(workers, flat.size());
    tbb::task_arena arena(workers);
    REQUIRE(arena.execute(
        [&] { return dt3_forced_parallel_build(owner, points); }));
    REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
    const dt3_parallel_view<Owner> view{owner};
    CHECK(dt3_canonical_cells(view) == expected);
    CHECK(dt3_spheres_are_empty<Int>(view));
    CHECK(owner._stats.parallel_insertions == owner._sites.size() - 4);
    CHECK(owner._stats.parallel_bands > 0);
  }
}

} // namespace

TEST_CASE("delaunay tets: parallel claims preserve exact complexes",
          "[topology][delaunay3][parallel]") {
  for (std::uint32_t seed = 0; seed < 8; ++seed) {
    CAPTURE(seed);
    dt3_check_parallel_coordinates<dt3_i32>(
        dt3_random_coordinates<dt3_i32>(seed, 40 + seed * 17, 200));
  }
  dt3_check_parallel_coordinates<dt3_i64>(
      dt3_random_coordinates<dt3_i64>(42, 100, 1000000000LL));
  auto scaled = dt3_random_coordinates<dt3_scaled_coord>(43, 100, 1000000000LL);
  for (auto &coordinate : scaled)
    coordinate *= 64;
  dt3_check_parallel_coordinates<dt3_i32>(scaled);

  std::vector<dt3_i32> grid;
  for (int x = 0; x < 5; ++x)
    for (int y = 0; y < 5; ++y)
      for (int z = 0; z < 5; ++z)
        for (int repeat = 0; repeat < 2; ++repeat)
          grid.insert(grid.end(), {x * 4, y * 4, z * 4});
  dt3_check_parallel_coordinates<dt3_i32>(grid);

  std::vector<dt3_i32> faces;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      faces.insert(faces.end(),
                   {i * 8, j * 8, 0, i * 8, 0, j * 8, 0, i * 8, j * 8});
  dt3_check_parallel_coordinates<dt3_i32>(faces);
}

TEST_CASE("delaunay tets: exhausted pools recover every uninserted site",
          "[topology][delaunay3][parallel]") {
  const auto flat = dt3_random_coordinates<dt3_i32>(273, 400, 10000);
  const auto points = dt3_lattice_points(flat);
  dt3_builder<> serial;
  REQUIRE(serial.build(points));
  for (std::size_t reserve :
       {std::size_t(0), std::size_t(1), std::size_t(16)}) {
    CAPTURE(reserve);
    dt3_owner owner;
    tbb::task_arena arena(8);
    REQUIRE(arena.execute(
        [&] { return dt3_forced_parallel_build(owner, points, reserve); }));
    CHECK(owner._stats.insertions == owner._sites.size() - 4);
    CHECK(dt3_owner_cells(owner) == dt3_canonical_cells(serial));
    REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
  }
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
  for (int workers : {1, 2, 4, 8, 16}) {
    CAPTURE(workers);
    tbb::task_arena arena(workers);
    REQUIRE(arena.execute([&] { return parallel.build(points); }));
    CHECK(parallel.is_valid());
    CHECK(dt3_canonical_cells(parallel) == expected);
    CHECK((parallel.stats().parallel_bands > 0) == (workers >= 4));
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
  using SmallOwner =
      tf::topology::cdt::dt3::tetrahedralization_owner<std::int8_t, dt3_i32,
                                                       dt3_i32>;
  const auto flat = dt3_random_coordinates<dt3_i32>(121, 60, 10000);
  SmallOwner owner;
  tbb::task_arena arena(8);
  CHECK(!arena.execute([&] {
    return dt3_forced_parallel_build(owner, dt3_lattice_points(flat));
  }));
  CHECK(owner._corners.size() == 0);
  CHECK(owner._refusal == tf::tetrahedralization_refusal::index_capacity);
}

TEST_CASE("delaunay tets: concurrent rows support wide index tickets",
          "[topology][delaunay3][parallel]") {
  const auto flat = dt3_random_coordinates<dt3_i32>(37, 512, 10000);
  const auto points = dt3_lattice_points(flat);
  dt3_builder<> reference;
  REQUIRE(reference.build(points));
  tf::topology::cdt::dt3::tetrahedralization_owner<std::int64_t, dt3_i32,
                                                   dt3_i32>
      owner;
  tbb::task_arena arena(8);
  REQUIRE(
      arena.execute([&] { return dt3_forced_parallel_build(owner, points); }));
  REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
  CHECK(dt3_canonical_cells(dt3_parallel_view<decltype(owner)>{owner}) ==
        dt3_canonical_cells(reference));
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
    const int count = 16 << ring;
    for (int i = 0; i < count; ++i)
      append((ring + 1) * .3, turn * (i + .25) / count);
  }
  return points;
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

auto dt3_contains_rim_edges(const dt3_owner &owner, int count) -> bool {
  std::vector<std::array<int, 2>> edges;
  for (std::size_t c = 0; c < owner._n_finite; ++c)
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j) {
        int a = owner._sites[std::size_t(owner._corners[c][i])].id;
        int b = owner._sites[std::size_t(owner._corners[c][j])].id;
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
  namespace dt3 = tf::topology::cdt::dt3;
  for (int height : {10, 10000}) {
    for (bool distributed : {false, true}) {
      auto flat = dt3_rim_with_interior_points(height);
      if (distributed) {
        std::vector<std::array<dt3_i32, 3>> interior;
        for (std::size_t i = 32; i < flat.size() / 3; ++i)
          interior.push_back({flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]});
        std::mt19937 random(731);
        std::shuffle(interior.begin(), interior.end(), random);
        for (std::size_t i = 0; i < interior.size(); ++i)
          for (std::size_t k = 0; k < 3; ++k)
            flat[3 * (i + 32) + k] = interior[i][k];
      }
      for (int workers : {1, 2, 4, 8, 16}) {
        CAPTURE(height, distributed, workers);
        dt3_owner owner;
        const auto rim =
            tf::make_points<3>(tf::make_range(flat.data(), 32 * 3));
        REQUIRE(dt3::build_tetrahedralization(owner, rim));
        REQUIRE(dt3_contains_rim_edges(owner, 32));
        std::size_t first = 32;
        tbb::task_arena arena(workers);
        for (std::size_t last :
             {std::size_t(49), std::size_t(81), std::size_t(145)}) {
          CAPTURE(first, last);
          dt3::tet_claim_workspace<int> work{};
          owner._order.clear();
          for (auto i = first; i < last; ++i) {
            const int site = int(owner._sites.size());
            owner._sites.push_back(
                {int(i), {flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]}});
            owner._order.push_back(site);
          }
          std::reverse(owner._order.begin(), owner._order.end());
          dt3::encode_tet_site_keys(owner, dt3::measure_tet_site_domain(owner));
          dt3::prepare_tet_claim_pool(owner, work, 2 * std::size_t(workers),
                                      owner._corners.size() +
                                          16 * (last - first));
          const auto inserted = owner._stats.parallel_insertions;
          REQUIRE(arena.execute([&] {
            dt3::insert_tet_claim_band(owner, work, 0, owner._order.size());
            return dt3::finish_tet_claims(owner, work);
          }));
          dt3::compact_tetrahedralization(owner);
          REQUIRE(owner._stats.parallel_insertions - inserted == last - first);
          REQUIRE(dt3::check_tetrahedralization_validity(owner));
          CHECK(dt3_contains_rim_edges(owner, 32));
          const dt3_parallel_view<dt3_owner> view{owner};
          CHECK(dt3_spheres_are_empty<dt3_i32>(view));
          tf::delaunay_tetrahedralizer<
              int, dt3_i32, dt3_i32,
              tf::topology::cdt::serial_delaunay_execution_policy>
              reference;
          REQUIRE(reference.build(
              tf::make_points<3>(tf::make_range(flat.data(), last * 3))));
          CHECK(dt3_named_cells(view) == dt3_named_cells(reference));
          first = last;
        }
      }
    }
  }
}

TEST_CASE(
    "delaunay tets: blocked cavities release claims without topology writes",
    "[topology][delaunay3][parallel]") {
  namespace dt3 = tf::topology::cdt::dt3;
  const auto flat = dt3_random_coordinates<dt3_i32>(712, 100, 10000);
  dt3_owner owner;
  REQUIRE(dt3::prepare_tet_claim_sites(owner, dt3_lattice_points(flat)));
  REQUIRE(dt3::bootstrap_tet_claims(owner, 0));
  dt3::tet_claim_workspace<int> work{};
  dt3::prepare_tet_claim_pool(owner, work, 2, owner._corners.size());
  dt3::tet_claim_scratch<int> scratch{};
  const auto site = owner._order[0];
  const auto seed = dt3::locate_claimed_tet(owner, work, 0, site, 0, scratch);
  REQUIRE(seed >= 0);
  const auto blocker = owner._neighbors[std::size_t(seed)][0];
  int interfering = -1;
  REQUIRE(dt3::acquire_tet_cell(work, 1, blocker, interfering));
  const auto corners = owner._corners;
  const auto neighbors = owner._neighbors;
  REQUIRE_FALSE(
      dt3::discover_claimed_tet_cavity(owner, work, 0, site, seed, scratch));
  dt3::release_tet_cells(work, 0, scratch.held);
  for (std::size_t i = 0; i < work.ownership.size(); ++i) {
    CHECK(work.ownership[i].load() ==
          (i == std::size_t(blocker) ? 1U : dt3::tet_unowned));
    for (std::size_t k = 0; k < 4; ++k) {
      CHECK(owner._corners[i][k] == corners[i][k]);
      CHECK(owner._neighbors[i][k] == neighbors[i][k]);
    }
  }
}

TEST_CASE("delaunay tets: edge pairing distinguishes high index bits",
          "[topology][delaunay3][parallel]") {
  namespace dt3 = tf::topology::cdt::dt3;
  using Index = std::int64_t;
  dt3::tetrahedralization_owner<Index, dt3_i32, dt3_i32> owner;
  owner._neighbors.allocate(4);
  dt3::tet_cavity_edge_table<Index> table{};
  table.generation = std::numeric_limits<std::uint32_t>::max();
  dt3::advance_tet_cavity_edges(table);
  REQUIRE(table.generation == 1);
  const Index high = Index(1) << 40;
  dt3::link_tet_cavity_edge(owner, table, Index(1), Index(2), Index(0), 0);
  dt3::link_tet_cavity_edge(owner, table, high + 1, high + 2, Index(1), 1);
  dt3::link_tet_cavity_edge(owner, table, Index(1), Index(2), Index(2), 2);
  dt3::link_tet_cavity_edge(owner, table, high + 1, high + 2, Index(3), 3);
  CHECK(owner._neighbors[0][0] == 2);
  CHECK(owner._neighbors[2][2] == 0);
  CHECK(owner._neighbors[1][1] == 3);
  CHECK(owner._neighbors[3][3] == 1);
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
          "[topology][delaunay3][append]") {
  for (int workers : {1, 4, 16}) {
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
    tbb::task_arena arena(workers);
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
    if (workers >= 4)
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

TEST_CASE(
    "delaunay tets: directed edge queries equal independent edge enumeration",
    "[topology][delaunay3][edge-query]") {
  dt3_builder<> dt;
  std::vector<dt3_i32> points;
  for (int x = 0; x < 5; ++x)
    for (int y = 0; y < 5; ++y)
      for (int z = 0; z < 5; ++z)
        points.insert(points.end(), {x * 64, y * 64, z});
  REQUIRE(dt.build(dt3_lattice_points(points)));
  std::vector<std::array<int, 2>> actual, queries;
  for (auto cell : dt.tets())
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j)
        actual.push_back(
            {std::min(cell[i], cell[j]), std::max(cell[i], cell[j])});
  std::sort(actual.begin(), actual.end());
  for (int i = 0; i < int(dt.n_sites()); ++i)
    for (int j = 0; j < int(dt.n_sites()); ++j)
      queries.push_back({i, j});
  for (int workers : {1, 4, 16}) {
    tbb::task_arena arena(workers);
    tf::topology::cdt::dt3::tet_edge_query_workspace<int> work;
    arena.execute([&] {
      tf::topology::cdt::dt3::state_missing_tet_edges(dt, queries, work);
    });
    std::vector<int> missing;
    for (std::size_t i = 0; i < queries.size(); ++i) {
      const auto q = queries[i];
      const bool present = std::binary_search(
          actual.begin(), actual.end(),
          std::array<int, 2>{std::min(q[0], q[1]), std::max(q[0], q[1])});
      REQUIRE(bool(work.present[i]) == present);
      if (!present)
        missing.push_back(int(i));
    }
    CHECK(std::vector<int>(work.missing.begin(), work.missing.end()) ==
          missing);
    work.present[1] = !work.present[1];
    CHECK(bool(work.present[1]) !=
          std::binary_search(actual.begin(), actual.end(), queries[1]));
  }
}

TEST_CASE(
    "delaunay tets: split admission states the earliest conflicting proposal",
    "[topology][delaunay3][edge-query]") {
  using Site = dt3_builder<>::site_type;
  std::vector<Site> standing{{9, {0, 0, 0}}, {12, {10, 0, 0}}};
  std::vector<Site> proposed{
      {20, {3, 0, 0}}, {21, {3, 0, 0}}, {22, {10, 0, 0}}};
  tf::buffer<tf::topology::cdt::dt3::tet_site_candidate<dt3_i32>> records;
  const auto first = [&] {
    return tf::topology::cdt::dt3::find_tet_site_coincidence(standing, proposed,
                                                             records);
  };
  CHECK(first() == 1);
  proposed[1].pt = tf::point<dt3_i32, 3>{4, 0, 0};
  CHECK(first() == 2);
  proposed[2].pt = tf::point<dt3_i32, 3>{5, 0, 0};
  CHECK(first() == proposed.size());
  proposed[0].pt = tf::point<dt3_i32, 3>{0, 0, 0};
  CHECK(first() == 0);
  proposed.clear();
  CHECK(first() == 0);
}

TEST_CASE(
    "delaunay tets: wide lattice sites append with exact midpoint proposals",
    "[topology][delaunay3][append]") {
  using DT = dt3_builder<dt3_i64>;
  using Site = DT::site_type;
  std::vector<Site> sites{{100, {-10, -10, -10}},
                          {101, {10, -10, -10}},
                          {102, {-10, 10, -10}},
                          {103, {-10, -10, 10}}};
  DT dt;
  REQUIRE(dt.build(tf::make_range(sites)));
  const std::array<std::array<int, 2>, 1> edges{
      {{dt.index_map().f()[0], dt.index_map().f()[1]}}};
  const std::array<int, 1> selected{0};
  tf::buffer<Site> proposed;
  tf::topology::cdt::dt3::propose_tet_edge_midpoints(dt, edges, selected, 104,
                                                     proposed);
  REQUIRE(proposed.size() == 1);
  CHECK(proposed[0].pt == tf::point<dt3_i64, 3>{0, -10, -10});
  REQUIRE(dt.append_sites(tf::make_range(proposed)));
  REQUIRE(dt.is_valid());
  CHECK(dt3_spheres_are_empty<dt3_i64>(dt));
  sites.push_back(proposed[0]);
  DT reference;
  REQUIRE(reference.build(tf::make_range(sites)));
  CHECK(dt3_named_cells(dt) == dt3_named_cells(reference));
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
  for (int count : {4095, 4096, 4097}) {
    CAPTURE(count);
    const auto flat = dt3_random_coordinates<dt3_i32>(731, count, 100000);
    const auto points = dt3_lattice_points(flat);
    Serial serial;
    REQUIRE(serial.build(points));
    const auto expected = dt3_canonical_cells(serial);
    for (int workers : {1, 4, 16}) {
      CAPTURE(workers);
      tbb::task_arena arena(workers);
      dt3_builder<> parallel;
      REQUIRE(arena.execute([&] { return parallel.build(points); }));
      CHECK(parallel.is_valid());
      CHECK(dt3_canonical_cells(parallel) == expected);
      CHECK((parallel.stats().parallel_bands > 0) ==
            (count >= 4096 && workers >= 4));
      CHECK(std::equal(serial.index_map().f().begin(),
                       serial.index_map().f().end(),
                       parallel.index_map().f().begin()));
    }
  }
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
          "[topology][delaunay3][parallel][determinism]") {
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
    for (int workers : {1, 2, 4, 16}) {
      CAPTURE(flat.size(), workers);
      tbb::task_arena arena(workers);
      dt3_builder<> parallel;
      REQUIRE(arena.execute([&] { return parallel.build(points); }));
      CHECK((parallel.stats().parallel_bands > 0) == (workers >= 4));
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
  for (int workers : {1, 2, 4, 16}) {
    CAPTURE(workers);
    tbb::task_arena arena(workers);
    dt3_builder<> parallel;
    REQUIRE(arena.execute([&] {
      return parallel.build(initial) && parallel.append_sites(appended);
    }));
    CHECK((parallel.stats().parallel_insertions > 0) == (workers >= 4));
    CHECK(dt3_published_stream(parallel) == expected);
  }
}

