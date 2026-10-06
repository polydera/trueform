/**
 * @file test_hole_table_tier.cpp
 * @brief Tests for the table tier: protection, candidates, the two stages
 *
 * Tests for:
 * - tf::fill::fill_table_hole
 * - tf::fill::solve_hole_table
 * - tf::fill::protect_hole_rim
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "fill_generators.hpp"
#include "oneapi/tbb/global_control.h"
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <tbb/task_arena.h>
#include <trueform/core/constants.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/fill/fill_holes.hpp>
#include <trueform/fill/holes/fill_table_hole.hpp>
#include <trueform/fill/holes/gather_hole_facets.hpp>
#include <trueform/fill/holes/hole_table.hpp>
#include <trueform/fill/holes/preflight_hole_rim.hpp>
#include <trueform/fill/holes/state_missing_rim_edges.hpp>
#include <trueform/fill/holes/validate_hole_rim.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/delaunay_tetrahedralizer.hpp>
#include <trueform/topology/face_membership.hpp>
#include <utility>
#include <vector>

namespace {

using table_index = int;
using table_int = tf::exact::int32;
using table_real = float;
using table_rim_t = tf::fill::hole_rim<table_index, table_int, table_real>;
using table_triangle = std::array<table_index, 3>;

/**
 * @brief The default objective of a patch stated in rim positions: the
 *        largest angle it charges and the area it covers. Every rim edge is
 *        charged against its carrying triangle and every interior chord
 *        against the two patch triangles that share it, each exactly once.
 */
auto table_objective(const table_rim_t &rim,
                     const std::vector<table_triangle> &patch)
    -> std::pair<double, double> {
  const table_index n = table_index(rim.size());
  const auto normal_of = [&rim](const table_triangle &triangle) {
    const int first = tf::fill::hole_canonical_rotation(
        rim.corners[std::size_t(triangle[0])],
        rim.corners[std::size_t(triangle[1])],
        rim.corners[std::size_t(triangle[2])]);
    return tf::fill::hole_oriented_normal(
        rim.positions[std::size_t(triangle[std::size_t(first)])]
            .template as<double>(),
        rim.positions[std::size_t(triangle[std::size_t((first + 1) % 3)])]
            .template as<double>(),
        rim.positions[std::size_t(triangle[std::size_t((first + 2) % 3)])]
            .template as<double>());
  };
  double angle = 0.0, area = 0.0;
  for (std::size_t i = 0; i < patch.size(); ++i) {
    const auto normal = normal_of(patch[i]);
    area += tf::fill::hole_triangle_area(normal);
    for (int e = 0; e < 3; ++e) {
      const table_index x = patch[i][std::size_t(e)];
      const table_index y = patch[i][std::size_t((e + 1) % 3)];
      if (table_index(y + 1 == n ? 0 : y + 1) == x) {
        angle = std::max(angle, tf::fill::hole_dihedral_angle(
                                    normal, rim.edges[std::size_t(y)].normal));
        continue;
      }
      for (std::size_t j = i + 1; j < patch.size(); ++j)
        for (int f = 0; f < 3; ++f)
          if (patch[j][std::size_t(f)] == y &&
              patch[j][std::size_t((f + 1) % 3)] == x)
            angle = std::max(angle, tf::fill::hole_dihedral_angle(
                                        normal, normal_of(patch[j])));
    }
  }
  return {angle, area};
}

/**
 * @brief Every triangulation of the rooted cycle over the admitted
 *        candidates, built interval by interval.
 */
auto table_enumerate(table_index i, table_index j,
                     const tf::buffer<table_triangle> &facets)
    -> std::vector<std::vector<table_triangle>> {
  if (j == i + 1)
    return {{}};
  std::vector<std::vector<table_triangle>> result;
  for (table_index k = i + 1; k < j; ++k) {
    const table_triangle key{i, k, j};
    if (!std::binary_search(facets.begin(), facets.end(), key))
      continue;
    const auto left = table_enumerate(i, k, facets);
    const auto right = table_enumerate(k, j, facets);
    for (const auto &first : left)
      for (const auto &second : right) {
        std::vector<table_triangle> patch{{j, k, i}};
        patch.insert(patch.end(), first.begin(), first.end());
        patch.insert(patch.end(), second.begin(), second.end());
        result.push_back(patch);
      }
  }
  return result;
}

/**
 * @brief The patch the filler produced, stated back in rim positions.
 */
auto table_patch_positions(const table_rim_t &rim,
                           const tf::buffer<table_triangle> &triangles)
    -> std::vector<table_triangle> {
  std::vector<table_triangle> patch;
  for (const auto &triangle : triangles) {
    table_triangle positions{-1, -1, -1};
    for (int corner = 0; corner < 3; ++corner)
      for (table_index k = 0; k < table_index(rim.size()); ++k)
        if (rim.corners[std::size_t(k)] == triangle[std::size_t(corner)])
          positions[std::size_t(corner)] = k;
    patch.push_back(positions);
  }
  return patch;
}

struct table_case {
  tf::polygons_buffer<table_index, table_real, 3, 3> mesh;
  tf::boundary_rims<table_index> rims;
};

using table_complex_t =
    tf::fill::hole_table_scratch<table_index, table_int, table_real>;

/**
 * @brief The complex over a prepared rim and the map from its sites back to
 *        rim positions, which both readings of the complex are taken
 *        against.
 */
auto table_prepare_complex(table_complex_t &complex, const table_rim_t &rim)
    -> bool {
  if (!complex.dt.build(tf::make_range(rim.sites)))
    return false;
  complex.site_of_position.allocate(rim.size());
  std::copy(complex.dt.index_map().f().begin(),
            complex.dt.index_map().f().end(), complex.site_of_position.begin());
  complex.position_of_site.allocate(complex.dt.n_sites());
  for (table_index k = 0; k < table_index(rim.size()); ++k)
    complex.position_of_site[std::size_t(
        complex.dt.index_map().f()[std::size_t(k)])] = k;
  return true;
}

/**
 * @brief The missing rim edges read the way the protection used to read
 *        them: every endpoint pair of every finite cell, sorted and
 *        deduplicated, then one binary search per rim edge.
 */
auto table_missing_by_edge_sort(const table_complex_t &complex, table_index n)
    -> std::vector<table_index> {
  std::vector<std::array<table_index, 2>> edges;
  for (auto tet : complex.dt.tets())
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j) {
        const table_index a =
            complex.position_of_site[std::size_t(tet[std::size_t(i)])];
        const table_index b =
            complex.position_of_site[std::size_t(tet[std::size_t(j)])];
        edges.push_back({std::min(a, b), std::max(a, b)});
      }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::vector<table_index> missing;
  for (table_index k = 0; k < n; ++k) {
    const table_index other = table_index(k + 1 == n ? 0 : k + 1);
    const std::array<table_index, 2> key{std::min(k, other),
                                         std::max(k, other)};
    if (!std::binary_search(edges.begin(), edges.end(), key))
      missing.push_back(k);
  }
  return missing;
}

/**
 * @brief The facets read the way the table used to read them: four per cell,
 *        sorted and deduplicated.
 */
auto table_facets_by_cell_emission(const table_complex_t &complex)
    -> std::vector<std::array<table_index, 3>> {
  std::vector<std::array<table_index, 3>> facets;
  for (auto tet : complex.dt.tets())
    for (int drop = 0; drop < 4; ++drop) {
      std::array<table_index, 3> facet{};
      int corner = 0;
      for (int slot = 0; slot < 4; ++slot)
        if (slot != drop)
          facet[std::size_t(corner++)] =
              complex.position_of_site[std::size_t(tet[std::size_t(slot)])];
      std::sort(facet.begin(), facet.end());
      facets.push_back(facet);
    }
  std::sort(facets.begin(), facets.end());
  facets.erase(std::unique(facets.begin(), facets.end()), facets.end());
  return facets;
}

auto table_random_case(std::mt19937 &rng, int n) -> table_case {
  std::uniform_real_distribution<float> jitter(-1.0f, 1.0f);
  std::vector<tf::point<table_real, 3>> rim;
  for (int k = 0; k < n; ++k) {
    const float angle = 6.2831853f * float(k) / float(n);
    const float radius = 1.0f + 0.6f * jitter(rng);
    rim.push_back({radius * std::cos(angle), radius * std::sin(angle),
                   0.7f * jitter(rng)});
  }
  auto mesh = hole_cone_mesh<table_index, table_real>(
      rim, tf::point<table_real, 3>{0.0f, 0.0f, -6.0f});
  auto rims = tf::make_boundary_rims(mesh.polygons());
  return {std::move(mesh), std::move(rims)};
}

/**
 * @brief A regular rim lifted by an alternating offset, whose cells meet in
 *        the near-ties a symmetric arrangement states.
 */
auto table_symmetric_case(int n, float lift) -> table_case {
  std::vector<tf::point<table_real, 3>> rim;
  for (int k = 0; k < n; ++k) {
    const float angle = 6.2831853f * float(k) / float(n);
    rim.push_back({std::cos(angle), std::sin(angle),
                   k % 2 == 0 ? lift : -lift});
  }
  auto mesh = hole_cone_mesh<table_index, table_real>(
      rim, tf::point<table_real, 3>{0.0f, 0.0f, -3.0f});
  auto rims = tf::make_boundary_rims(mesh.polygons());
  return {std::move(mesh), std::move(rims)};
}

/**
 * @brief A long rim edge with a collar of rim points standing around its
 *        middle. Every ball through that edge's ends reaches the collar, so
 *        the complex cannot hold the edge and the protection must split it.
 */
auto table_shadowed_case(int collar) -> table_case {
  std::vector<tf::point<table_real, 3>> rim{{-10.0f, 0.0f, 0.0f},
                                            {10.0f, 0.0f, 0.0f}};
  for (int k = 0; k < collar; ++k) {
    const float angle = 6.2831853f * float(k) / float(collar);
    rim.push_back({0.0f, std::cos(angle), std::sin(angle)});
  }
  auto mesh = hole_cone_mesh<table_index, table_real>(
      rim, tf::point<table_real, 3>{0.0f, 0.0f, -20.0f});
  auto rims = tf::make_boundary_rims(mesh.polygons());
  return {std::move(mesh), std::move(rims)};
}

/**
 * @brief A square rim around a raised fifth point, which preflight prepares
 *        as a simple cycle.
 */
auto table_square_case() -> table_case {
  std::vector<tf::point<table_real, 3>> rim{{0.0f, 0.0f, 0.0f},
                                            {4.0f, 0.0f, 0.0f},
                                            {4.0f, 4.0f, 0.0f},
                                            {0.0f, 4.0f, 0.0f},
                                            {2.0f, 2.0f, 3.0f}};
  auto mesh = hole_cone_mesh<table_index, table_real>(
      rim, tf::point<table_real, 3>{2.0f, 2.0f, -4.0f});
  auto rims = tf::make_boundary_rims(mesh.polygons());
  return {std::move(mesh), std::move(rims)};
}

/**
 * @brief Stand site 2 on the midpoint of rim edge 0, so the complex cannot
 *        hold that edge and its midpoint is a site already there. Preflight
 *        never prepares such a rim; only the protection's refusals read it.
 */
auto table_block_first_edge(table_rim_t &rim) -> void {
  using coord_t = tf::fill::hole_site_coord<table_int>;
  for (std::size_t axis = 0; axis < 3; ++axis)
    rim.sites[2].pt[axis] =
        (rim.sites[0].pt[axis] + rim.sites[1].pt[axis]) / coord_t(2);
}

auto table_prepare_rim(const table_case &instance,
                       tf::face_membership<table_index> &membership,
                       tf::fill::hole_rim_scratch<table_index, table_int>
                           &preflight,
                       table_rim_t &rim) -> bool {
  if (instance.rims.size() != 1)
    return false;
  membership.build(instance.mesh.polygons());
  const auto converter = tf::exact::make_pt_converter<table_int, table_real>(
      instance.mesh.points());
  const auto verdict = tf::fill::preflight_hole_rim(
      instance.mesh.polygons(), membership, instance.rims, 0, converter,
      preflight, rim);
  return verdict.status == tf::hole_fill_status::filled;
}

/**
 * @brief One rim solved end to end, leaving the table it was solved on.
 */
auto table_solve_lanes(const table_case &instance,
                       tf::fill::hole_table<table_index> &table,
                       tf::buffer<table_triangle> &triangles)
    -> tf::hole_fill_status {
  tf::face_membership<table_index> membership;
  tf::fill::hole_rim_scratch<table_index, table_int> preflight;
  table_rim_t rim;
  if (!table_prepare_rim(instance, membership, preflight, rim))
    return tf::hole_fill_status::refused_invalid_rim;
  table_complex_t complex;
  tf::buffer<tf::point<table_real, 3>> minted;
  tf::buffer<tf::hole_split<table_index>> splits;
  table_index offending = -1;
  return tf::fill::fill_table_hole(
      instance.mesh.polygons(), membership,
      table_index(instance.mesh.points().size()), complex, table, rim,
      triangles, minted, splits, offending);
}

/**
 * @brief Whether two lanes hold the same bytes, element by element.
 */
template <typename Stated, typename Reference>
auto table_lane_holds(const Stated &stated, const Reference &reference,
                      std::size_t count) -> bool {
  for (std::size_t k = 0; k < count; ++k)
    if (std::memcmp(&stated[k], &reference[k], sizeof(reference[k])) != 0)
      return false;
  return true;
}

} // namespace

TEST_CASE("fill: the table states the exact lexicographic optimum",
          "[fill][table]") {
  std::mt19937 rng(20260922u);
  int compared = 0;
  for (int trial = 0; trial < 120; ++trial) {
    const int n = 4 + int(rng() % 5);
    auto instance = table_random_case(rng, n);
    if (instance.rims.size() != 1)
      continue;

    tf::face_membership<table_index> membership;
    membership.build(instance.mesh.polygons());
    const auto converter =
        tf::exact::make_pt_converter<table_int, table_real>(
            instance.mesh.points());
    tf::fill::hole_rim_scratch<table_index, table_int> preflight;
    table_rim_t rim;
    const auto verdict = tf::fill::preflight_hole_rim(
        instance.mesh.polygons(), membership, instance.rims, 0, converter,
        preflight, rim);
    if (verdict.status != tf::hole_fill_status::filled)
      continue;

    tf::fill::hole_table_scratch<table_index, table_int, table_real> complex;
    tf::fill::hole_table<table_index> table;
    tf::buffer<table_triangle> triangles;
    tf::buffer<tf::point<table_real, 3>> minted;
    tf::buffer<tf::hole_split<table_index>> splits;
    table_index offending = -1;
    const auto status = tf::fill::fill_table_hole(
        instance.mesh.polygons(), membership,
        table_index(instance.mesh.points().size()), complex, table, rim,
        triangles, minted, splits, offending);
    if (status != tf::hole_fill_status::filled)
      continue;

    const auto patch = table_patch_positions(rim, triangles);
    const auto stated = table_objective(rim, patch);
    const auto candidates =
        table_enumerate(0, table_index(rim.size()) - 1, table.facets);
    REQUIRE(!candidates.empty());
    auto best = table_objective(rim, candidates.front());
    for (const auto &candidate : candidates) {
      const auto value = table_objective(rim, candidate);
      if (value.first < best.first ||
          (value.first == best.first && value.second < best.second))
        best = value;
    }
    REQUIRE(stated.first == best.first);
    REQUIRE(stated.second <= best.second * (1.0 + 1e-6));
    REQUIRE(patch.size() == std::size_t(rim.size()) - 2);
    ++compared;
  }
  REQUIRE(compared >= 60);
}

TEST_CASE("fill: a root whose every candidate is forbidden refuses",
          "[fill][table]") {
  const auto plane = tf::make_plane_mesh<table_index>(4.0f, 4.0f, 3, 3);
  const auto rims = tf::make_boundary_rims(plane.polygons());
  REQUIRE(rims.size() == 1);
  const auto result = tf::fill_holes(plane.polygons(), rims);
  REQUIRE(result.size() == 1);
  REQUIRE(result.status[0] == tf::hole_fill_status::refused_table);
  REQUIRE(result.triangles[0].size() == 0);
  REQUIRE(result.offending[0][0] == -1);
  REQUIRE(result.offending[0][1] == -1);
}

TEST_CASE("fill: a protection split that lands on a site refuses the rim",
          "[fill][table]") {
  const auto instance = table_square_case();
  tf::face_membership<table_index> membership;
  tf::fill::hole_rim_scratch<table_index, table_int> preflight;
  table_rim_t rim;
  REQUIRE(table_prepare_rim(instance, membership, preflight, rim));
  table_block_first_edge(rim);

  tf::fill::hole_table_scratch<table_index, table_int, table_real> complex;
  table_index offending = -1;
  const auto status = tf::fill::protect_hole_rim(instance.mesh.points(),
                                                 complex, rim, offending);
  REQUIRE(status == tf::hole_fill_status::refused_invalid_rim);
  REQUIRE(offending == 0);
}

TEST_CASE("fill: a rim edge still absent at the deepest split refuses",
          "[fill][table]") {
  const auto instance = table_square_case();
  tf::face_membership<table_index> membership;
  tf::fill::hole_rim_scratch<table_index, table_int> preflight;
  table_rim_t rim;
  REQUIRE(table_prepare_rim(instance, membership, preflight, rim));
  table_block_first_edge(rim);
  auto &edge = rim.edges[0];
  edge.t1 = std::uint8_t(edge.t0 == 0 ? 1 : edge.t0 - 1);

  tf::fill::hole_table_scratch<table_index, table_int, table_real> complex;
  table_index offending = -1;
  const auto status = tf::fill::protect_hole_rim(instance.mesh.points(),
                                                 complex, rim, offending);
  REQUIRE(status == tf::hole_fill_status::refused_protection_depth);
  REQUIRE(offending == 0);
}

TEST_CASE("fill: the presence scan states the edge table's own missing list",
          "[fill][table]") {
  std::mt19937 rng(20260923u);
  int compared = 0, missing_seen = 0, split_seen = 0;
  for (int trial = 0; trial < 80; ++trial) {
    const int n = 5 + int(rng() % 9);
    const auto instance = trial % 4 == 0 ? table_shadowed_case(3 + trial % 6)
                          : trial % 4 == 1
                              ? table_symmetric_case(n, 0.08f * float(n % 5))
                              : table_random_case(rng, n);
    tf::face_membership<table_index> membership;
    tf::fill::hole_rim_scratch<table_index, table_int> preflight;
    table_rim_t rim;
    if (!table_prepare_rim(instance, membership, preflight, rim))
      continue;

    table_complex_t complex;
    if (!table_prepare_complex(complex, rim))
      continue;
    const table_index before = table_index(rim.size());
    tf::fill::state_missing_rim_edges(complex.dt, complex.site_of_position,
                                      complex.edge_query);
    const std::vector<table_index> stated(complex.edge_query.missing.begin(),
                                          complex.edge_query.missing.end());
    CHECK(stated == table_missing_by_edge_sort(complex, before));
    if (!stated.empty())
      ++missing_seen;

    table_index offending = -1;
    if (tf::fill::protect_hole_rim(instance.mesh.points(), complex, rim,
                                   offending) != tf::hole_fill_status::filled)
      continue;
    if (table_index(rim.size()) > before)
      ++split_seen;

    // The protected rim carries its splits as sites of their own, and the
    // complex over it holds every one of its edges.
    REQUIRE(table_prepare_complex(complex, rim));
    const table_index after = table_index(rim.size());
    tf::fill::state_missing_rim_edges(complex.dt, complex.site_of_position,
                                      complex.edge_query);
    const std::vector<table_index> protected_rim(
        complex.edge_query.missing.begin(), complex.edge_query.missing.end());
    CHECK(protected_rim == table_missing_by_edge_sort(complex, after));
    CHECK(protected_rim.empty());
    ++compared;
  }
  CHECK(compared >= 40);
  CHECK(missing_seen > 0);
  CHECK(split_seen > 0);
}

TEST_CASE("fill: a rim edge no cell can carry is named, the wrap included",
          "[fill][table]") {
  // A site standing on the open segment of a rim edge keeps that edge out of
  // every cell, which names a missing edge without relying on any rounding.
  using table_coord = tf::fill::hole_site_coord<table_int>;
  const std::vector<tf::point<table_real, 3>> points{{0.0f, 0.0f, 0.0f},
                                                     {4.0f, 0.0f, 0.0f},
                                                     {5.0f, 4.0f, 1.0f},
                                                     {2.0f, 5.0f, 0.0f},
                                                     {-1.0f, 3.0f, 2.0f}};
  const auto mesh = hole_cone_mesh<table_index, table_real>(
      points, tf::point<table_real, 3>{2.0f, 2.0f, -4.0f});
  const table_case instance{mesh, tf::make_boundary_rims(mesh.polygons())};

  for (int wrapping = 0; wrapping < 2; ++wrapping) {
    tf::face_membership<table_index> membership;
    tf::fill::hole_rim_scratch<table_index, table_int> preflight;
    table_rim_t rim;
    REQUIRE(table_prepare_rim(instance, membership, preflight, rim));
    const table_index n = table_index(rim.size());
    const table_index blocked = wrapping ? table_index(n - 1) : table_index(0);
    const table_index from = blocked;
    const table_index to = table_index(blocked + 1 == n ? 0 : blocked + 1);
    for (std::size_t axis = 0; axis < 3; ++axis)
      rim.sites[2].pt[axis] = (rim.sites[std::size_t(from)].pt[axis] +
                               rim.sites[std::size_t(to)].pt[axis]) /
                              table_coord(2);

    table_complex_t complex;
    REQUIRE(table_prepare_complex(complex, rim));
    tf::fill::state_missing_rim_edges(complex.dt, complex.site_of_position,
                                      complex.edge_query);
    const std::vector<table_index> stated(complex.edge_query.missing.begin(),
                                          complex.edge_query.missing.end());
    CHECK(stated == table_missing_by_edge_sort(complex, n));
    CHECK(std::find(stated.begin(), stated.end(), blocked) != stated.end());
  }
}

TEST_CASE("fill: the table lanes are the same at any worker count",
          "[fill][table]") {
  // A rim whose widest span band outgrows the band grain, so the schedule
  // under test is the parallel one and not its serial fallback.
  std::vector<tf::point<table_real, 3>> ring;
  for (int k = 0; k < 4096; ++k) {
    const float angle = 6.2831853f * float(k) / 4096.0f;
    ring.push_back({std::cos(angle), std::sin(angle),
                    0.35f * std::sin(3.0f * angle)});
  }
  const auto mesh = hole_cone_mesh<table_index, table_real>(
      ring, tf::point<table_real, 3>{0.0f, 0.0f, -1.5f});
  const table_case instance{mesh, tf::make_boundary_rims(mesh.polygons())};
  REQUIRE(instance.rims.size() == 1);

  tf::fill::hole_table<table_index> reference;
  tf::buffer<table_triangle> reference_triangles;
  REQUIRE(table_solve_lanes(instance, reference, reference_triangles) ==
          tf::hole_fill_status::filled);
  const std::size_t kept = reference.facets.size();
  const std::size_t states = 2 * kept + 1;
  std::size_t widest = 0;
  for (std::size_t span = 0; span < reference.bands.size(); ++span)
    widest = std::max(widest, reference.bands[span].size());
  REQUIRE(widest > tf::fill::hole_table_band_grain);

  for (std::size_t workers : {std::size_t(1), std::size_t(2), std::size_t(4),
                              std::size_t(16)}) {
    const tbb::global_control control(
        tbb::global_control::max_allowed_parallelism, workers);
    tf::fill::hole_table<table_index> stated;
    tf::buffer<table_triangle> triangles;
    REQUIRE(table_solve_lanes(instance, stated, triangles) ==
            tf::hole_fill_status::filled);
    REQUIRE(stated.facets.size() == kept);
    CHECK(table_lane_holds(stated.facets, reference.facets, kept));
    CHECK(table_lane_holds(stated.normal, reference.normal, kept));
    CHECK(table_lane_holds(stated.area, reference.area, kept));
    CHECK(table_lane_holds(stated.boundary, reference.boundary, kept));
    CHECK(table_lane_holds(stated.bottleneck, reference.bottleneck, states));
    CHECK(table_lane_holds(stated.total, reference.total, states));
    CHECK(table_lane_holds(stated.choice, reference.choice, states));
    CHECK(table_lane_holds(stated.valid, reference.valid, states));
    REQUIRE(triangles.size() == reference_triangles.size());
    CHECK(table_lane_holds(triangles, reference_triangles, triangles.size()));
  }
}

TEST_CASE("fill: the facet a cell owns is the four-per-cell set",
          "[fill][table]") {
  std::mt19937 rng(20260924u);
  int compared = 0;
  std::size_t hull_facets = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const int n = 5 + int(rng() % 9);
    const auto instance = trial % 4 == 0 ? table_shadowed_case(3 + trial % 6)
                          : trial % 4 == 1
                              ? table_symmetric_case(n, 0.05f * float(n % 4))
                              : table_random_case(rng, n);
    tf::face_membership<table_index> membership;
    tf::fill::hole_rim_scratch<table_index, table_int> preflight;
    table_rim_t rim;
    if (!table_prepare_rim(instance, membership, preflight, rim))
      continue;

    table_complex_t complex;
    table_index offending = -1;
    if (tf::fill::protect_hole_rim(instance.mesh.points(), complex, rim,
                                   offending) != tf::hole_fill_status::filled)
      continue;
    REQUIRE(table_prepare_complex(complex, rim));

    tf::buffer<std::array<table_index, 3>> facets;
    tf::buffer<table_index> owned;
    tf::fill::gather_hole_facets(complex.dt, complex.position_of_site, owned,
                                 facets);
    const std::vector<std::array<table_index, 3>> stated(facets.begin(),
                                                         facets.end());
    CHECK(stated == table_facets_by_cell_emission(complex));
    REQUIRE(!stated.empty());
    for (auto row : complex.dt.neighbors())
      for (std::size_t slot = 0; slot < 4; ++slot)
        if (row[slot] == complex.dt.k_none)
          ++hull_facets;
    ++compared;
  }
  CHECK(compared >= 30);
  // A rim's complex is a convex hull with a boundary, so the hull-owned
  // facets are exercised by every case.
  CHECK(hull_facets > 0);
}

TEST_CASE("hole table: parallel Delaunay preserves rim candidates",
          "[fill][hole_table][parallel]") {
  using Serial = tf::delaunay_tetrahedralizer<
      int, table_int, table_int,
      tf::topology::cdt::serial_delaunay_execution_policy>;
  constexpr int n = 16384;
  std::vector<table_int> coordinates;
  coordinates.reserve(3 * n);
  for (int i = 0; i < n; ++i) {
    const double angle = 2 * tf::pi<double> * double(i) / double(n);
    coordinates.push_back(table_int(std::llround(1000000 * std::cos(angle))));
    coordinates.push_back(table_int(std::llround(1000000 * std::sin(angle))));
    coordinates.push_back(table_int(std::llround(300000 * std::sin(3 * angle))));
  }
  const auto points = tf::make_points<3>(tf::make_range(coordinates));
  Serial serial;
  tf::delaunay_tetrahedralizer<int, table_int> parallel;
  REQUIRE(serial.build(points));
  tbb::task_arena arena(8);
  REQUIRE(arena.execute([&] { return parallel.build(points); }));
  REQUIRE(parallel.stats().parallel_insertions > 0);
  REQUIRE(parallel.n_sites() == n);
  tf::buffer<int> serial_positions, parallel_positions;
  serial_positions.allocate(serial.n_sites());
  parallel_positions.allocate(parallel.n_sites());
  for (int i = 0; i < n; ++i) {
    serial_positions[std::size_t(serial.index_map().f()[std::size_t(i)])] = i;
    parallel_positions[std::size_t(parallel.index_map().f()[std::size_t(i)])] = i;
  }
  tf::buffer<table_triangle> serial_facets, parallel_facets;
  tf::buffer<int> serial_owned, parallel_owned, serial_missing,
      parallel_missing;
  tf::topology::cdt::dt3::tet_edge_query_workspace<int> serial_query,
      parallel_query;
  tf::fill::gather_hole_facets(serial, serial_positions, serial_owned,
                              serial_facets);
  tf::fill::gather_hole_facets(parallel, parallel_positions, parallel_owned,
                              parallel_facets);
  REQUIRE(serial_facets.size() > 0);
  REQUIRE(serial_facets.size() == parallel_facets.size());
  CHECK(std::equal(serial_facets.begin(), serial_facets.end(),
                   parallel_facets.begin()));
  tf::fill::state_missing_rim_edges(serial, serial.index_map().f(),
                                    serial_query);
  std::swap(serial_missing, serial_query.missing);
  tf::fill::state_missing_rim_edges(parallel, parallel.index_map().f(),
                                    parallel_query);
  std::swap(parallel_missing, parallel_query.missing);
  REQUIRE(serial_missing.size() == parallel_missing.size());
  CHECK(std::equal(serial_missing.begin(), serial_missing.end(),
                   parallel_missing.begin()));
}
