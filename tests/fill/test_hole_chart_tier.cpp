/**
 * @file test_hole_chart_tier.cpp
 * @brief Tests for the chart tier inside the fill's ladder
 *
 * Tests for:
 * - tf::fill_holes through tf::fill::fill_chart_hole
 * - tf::fill::make_hole_chart_patch / tf::fill::emit_hole_chart_patch
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/exact/pt_converter.hpp>
#include <trueform/fill/fill_holes.hpp>
#include <trueform/fill/hole_fill_result.hpp>
#include <trueform/fill/hole_fill_status.hpp>
#include <trueform/fill/make_filled_mesh.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/topology/boundary_rims.hpp>
#include <trueform/topology/is_manifold.hpp>
#include "fill_generators.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

using tier_index = int;
using tier_real = double;
using tier_result = tf::hole_fill_result<tier_index, tier_real>;

/// The worst distance of a group's mints from the unit sphere.
auto tier_sphere_error(const tier_result &result, std::size_t group) -> double {
  double worst = 0.0;
  for (const auto &p : result.minted_points[group])
    worst = std::max(
        worst,
        std::abs(std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - 1.0));
  return worst;
}

/// Whether every corner of a group's patch is a rim vertex or one of its own
/// mints, and every mint is a corner.
auto tier_names_rim_and_mints(const tier_result &result, std::size_t group,
                              const std::vector<tier_index> &rim,
                              tier_index n_points) -> bool {
  std::vector<char> used(result.minted_points[group].size(), 0);
  for (const auto &triangle : result.triangles[group])
    for (auto corner : triangle) {
      if (corner >= n_points) {
        const std::size_t mint = std::size_t(corner - n_points);
        if (mint >= used.size())
          return false;
        used[mint] = 1;
      } else if (std::find(rim.begin(), rim.end(), corner) == rim.end())
        return false;
    }
  return std::all_of(used.begin(), used.end(), [](char u) { return u != 0; });
}

/// The plane z = 0 over an 8 x 8 grid with its central 2 x 2 cells removed,
/// every cell two triangles but `quad`, which stays one quad.
auto tier_quad_grid(std::size_t quad)
    -> tf::polygons_buffer<tier_index, tier_real, 3, tf::dynamic_size> {
  tf::polygons_buffer<tier_index, tier_real, 3, tf::dynamic_size> mesh;
  mesh.points_buffer().allocate(81);
  for (int j = 0; j <= 8; ++j)
    for (int i = 0; i <= 8; ++i)
      mesh.points()[std::size_t(j * 9 + i)] =
          tf::point<tier_real, 3>{tier_real(i), tier_real(j), 0.0};
  auto &offsets = mesh.faces_buffer().offsets_buffer();
  auto &corners = mesh.faces_buffer().data_buffer();
  offsets.push_back(0);
  const auto face = [&](std::vector<tier_index> ids) {
    for (auto id : ids)
      corners.push_back(id);
    offsets.push_back(tier_index(corners.size()));
  };
  for (int j = 0; j < 8; ++j)
    for (int i = 0; i < 8; ++i) {
      if (i >= 3 && i < 5 && j >= 3 && j < 5)
        continue;
      const tier_index a = tier_index(j * 9 + i), b = a + 1, c = a + 9,
                       d = c + 1;
      if (std::size_t(j * 8 + i) == quad)
        face({a, b, d, c});
      else {
        face({a, b, d});
        face({a, d, c});
      }
    }
  return mesh;
}

auto tier_rim_vertices(const tf::boundary_rims<tier_index> &rims)
    -> std::vector<tier_index> {
  std::vector<tier_index> rim;
  for (auto vertex : rims.vertices[0])
    rim.push_back(vertex);
  return rim;
}

} // namespace

TEST_CASE("fill: the chart tier continues a spherical cap", "[fill][chart]") {
  const auto band = hole_sphere_band_mesh<tier_index, tier_real>(72, 50.0, 4);
  const auto rims = hole_rim_holding(tf::make_boundary_rims(band.polygons()),
                                     tier_index(0));
  REQUIRE(rims.size() == 1);
  const auto result = tf::fill_holes(band.polygons(), rims);

  REQUIRE(result.status[0] == tf::hole_fill_status::filled);
  REQUIRE(result.charted[0] == tf::hole_chart_status::admitted);
  REQUIRE(result.refined[0] != tf::hole_refine_status::not_attempted);
  REQUIRE(result.faired[0] == tf::hole_fair_status::not_attempted);
  REQUIRE(result.splits.size() == 0);
  REQUIRE(result.plan_faces.size() == 0);
  REQUIRE(result.minted_points[0].size() > 0);
  REQUIRE(tier_names_rim_and_mints(result, 0, tier_rim_vertices(rims),
                                   tier_index(band.points().size())));
  REQUIRE(tier_sphere_error(result, 0) < 1e-3);
  REQUIRE(result.seam_max_angle[0] >= 0.0);
  REQUIRE(result.seam_max_angle[0] < 0.2);

  const auto filled = tf::make_filled_mesh(band.polygons(), result);
  REQUIRE(tf::is_manifold(filled.polygons()));
}

TEST_CASE("fill: the chart's residual holds its rim alike at any scale",
          "[fill][chart]") {
  const auto unit =
      hole_sphere_band_mesh<tier_index, tier_real>(72, 50.0, 4, 0.01);
  auto tiny = unit;
  const double scale = 1e-14;
  for (std::size_t i = 0; i < tiny.points().size(); ++i)
    for (std::size_t axis = 0; axis < 3; ++axis)
      tiny.points()[i][axis] *= scale;
  const auto unit_result = tf::fill_holes(
      unit.polygons(),
      hole_rim_holding(tf::make_boundary_rims(unit.polygons()), tier_index(0)));
  const auto tiny_result = tf::fill_holes(
      tiny.polygons(),
      hole_rim_holding(tf::make_boundary_rims(tiny.polygons()), tier_index(0)));
  REQUIRE(unit_result.charted[0] == tf::hole_chart_status::admitted);
  REQUIRE(tiny_result.charted[0] == tf::hole_chart_status::admitted);
  INFO("seam unit " << unit_result.seam_max_angle[0] << " tiny "
                    << tiny_result.seam_max_angle[0]);
  REQUIRE(tiny_result.seam_max_angle[0] <
          1.02 * unit_result.seam_max_angle[0]);
}

TEST_CASE("fill: a planar host fills through the chart in its own plane",
          "[fill][chart]") {
  const auto plane = tf::make_plane_mesh<tier_index>(4.0, 4.0, 12, 12);
  std::vector<std::size_t> cells;
  for (std::size_t j = 3; j < 9; ++j)
    for (std::size_t i = 3; i < 9; ++i) {
      cells.push_back(2 * (12 * j + i));
      cells.push_back(2 * (12 * j + i) + 1);
    }
  const auto holed = hole_punched_mesh(plane, cells);
  const auto boundary = tf::make_boundary_rims(holed.polygons());
  REQUIRE(boundary.size() == 2);
  const std::size_t hole =
      boundary.vertices[0].size() < boundary.vertices[1].size() ? 0 : 1;
  const auto rims = hole_turned_rim(boundary, tier_index(hole), tier_index(0),
                                    false);
  REQUIRE(rims.vertices[0].size() > 5);
  const auto result = tf::fill_holes(holed.polygons(), rims);

  REQUIRE(result.status[0] == tf::hole_fill_status::filled);
  REQUIRE(result.charted[0] == tf::hole_chart_status::admitted);
  REQUIRE(result.splits.size() == 0);
  REQUIRE(result.minted_points[0].size() > 0);
  REQUIRE(tier_names_rim_and_mints(result, 0, tier_rim_vertices(rims),
                                   tier_index(holed.points().size())));
  for (const auto &p : result.minted_points[0])
    REQUIRE(std::abs(p[2]) <= 1e-12);
  REQUIRE(result.seam_max_angle[0] <= 1e-9);
}

TEST_CASE("fill: an exactly planar rim the chart refuses is lidded in its "
          "own plane",
          "[fill][chart]") {
  const double pi = 3.14159265358979323846;
  for (double polar : {85.0, 105.0}) {
    const auto band =
        hole_sphere_band_mesh<tier_index, tier_real>(72, polar, 4);
    const auto rims = hole_rim_holding(tf::make_boundary_rims(band.polygons()),
                                       tier_index(0));
    REQUIRE(rims.size() == 1);
    const auto result = tf::fill_holes(band.polygons(), rims);
    REQUIRE(result.status[0] == tf::hole_fill_status::filled);
    REQUIRE(result.charted[0] == tf::hole_chart_status::rim_plane);
    REQUIRE(result.refined[0] != tf::hole_refine_status::not_attempted);
    REQUIRE(result.faired[0] == tf::hole_fair_status::not_attempted);
    REQUIRE(result.splits.size() == 0);
    REQUIRE(result.minted_points[0].size() > 0);
    REQUIRE(tier_names_rim_and_mints(result, 0, tier_rim_vertices(rims),
                                     tier_index(band.points().size())));
    const double level = std::cos(polar * pi / 180.0);
    for (const auto &p : result.minted_points[0])
      REQUIRE(std::abs(p[2] - level) <= 1e-12);
    const auto filled = tf::make_filled_mesh(band.polygons(), result);
    REQUIRE(tf::is_manifold(filled.polygons()));
  }
}

TEST_CASE("fill: an exactly planar rim whose collar holds a quad is lidded",
          "[fill][chart]") {
  for (std::size_t quad : {std::size_t(2 * 8 + 2), std::size_t(1 * 8 + 1)}) {
    const auto mesh = tier_quad_grid(quad);
    const auto boundary = tf::make_boundary_rims(mesh.polygons());
    REQUIRE(boundary.size() == 2);
    const std::size_t hole =
        boundary.vertices[0].size() < boundary.vertices[1].size() ? 0 : 1;
    const auto rims =
        hole_turned_rim(boundary, tier_index(hole), tier_index(0), false);
    REQUIRE(rims.vertices[0].size() == 8);
    const auto result = tf::fill_holes(mesh.polygons(), rims);
    REQUIRE(result.status[0] == tf::hole_fill_status::filled);
    REQUIRE(result.charted[0] == tf::hole_chart_status::rim_plane);
    REQUIRE(result.splits.size() == 0);
    for (const auto &p : result.minted_points[0])
      REQUIRE(p[2] == 0.0);
    const auto filled = tf::make_filled_mesh(mesh.polygons(), result);
    REQUIRE(tf::is_manifold(filled.polygons()));
  }
}

TEST_CASE("fill: a refused chart off a plane hands its rim to the table",
          "[fill][chart]") {
  const auto steep =
      hole_sphere_band_mesh<tier_index, tier_real>(72, 85.0, 4, 0.01);
  const auto folded =
      hole_sphere_band_mesh<tier_index, tier_real>(72, 105.0, 4, 0.01);
  for (const auto *band : {&steep, &folded}) {
    const auto rims = hole_rim_holding(
        tf::make_boundary_rims(band->polygons()), tier_index(0));
    REQUIRE(rims.size() == 1);
    const auto result = tf::fill_holes(band->polygons(), rims);
    REQUIRE(result.charted[0] == (band == &steep
                                      ? tf::hole_chart_status::steep
                                      : tf::hole_chart_status::folded));
    REQUIRE(result.status[0] == tf::hole_fill_status::filled);
    REQUIRE(result.triangles[0].size() >= rims.vertices[0].size() - 2);
    const auto filled = tf::make_filled_mesh(band->polygons(), result);
    REQUIRE(tf::is_manifold(filled.polygons()));
  }
}

TEST_CASE("fill: a lid joining rim vertices the host joins is refused and "
          "reported where the table refuses too",
          "[fill][chart]") {
  const auto plane = tf::make_plane_mesh<tier_index>(4.0, 4.0, 3, 3);
  const auto rims = tf::make_boundary_rims(plane.polygons());
  REQUIRE(rims.size() == 1);
  const auto result = tf::fill_holes(plane.polygons(), rims);
  REQUIRE(result.status[0] == tf::hole_fill_status::refused_table);
  REQUIRE(result.charted[0] == tf::hole_chart_status::forbidden_chord);
  REQUIRE(result.refined[0] == tf::hole_refine_status::not_attempted);
}

TEST_CASE("fill: a rim the enumeration holds never reads a chart",
          "[fill][chart]") {
  const auto mesh = hole_cone_mesh<tier_index, tier_real>(
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.1}, {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.1}},
      tf::point<tier_real, 3>{0.0, 0.0, -2.0});
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  REQUIRE(rims.size() == 1);
  const auto result = tf::fill_holes(mesh.polygons(), rims);
  REQUIRE(result.status[0] == tf::hole_fill_status::filled);
  REQUIRE(result.charted[0] == tf::hole_chart_status::not_attempted);
}

TEST_CASE("fill: a rim of twelve hundred vertices samples by its rim",
          "[fill][chart]") {
  const auto band =
      hole_sphere_band_mesh<tier_index, tier_real>(1200, 50.0, 6);
  const auto rims = hole_rim_holding(tf::make_boundary_rims(band.polygons()),
                                     tier_index(0));
  REQUIRE(rims.size() == 1);
  const auto result = tf::fill_holes(band.polygons(), rims);

  REQUIRE(result.status[0] == tf::hole_fill_status::filled);
  REQUIRE(result.charted[0] == tf::hole_chart_status::admitted);
  REQUIRE(result.minted_points[0].size() > 0);
  REQUIRE(result.minted_points[0].size() < 4 * rims.vertices[0].size());
  REQUIRE(tier_names_rim_and_mints(result, 0, tier_rim_vertices(rims),
                                   tier_index(band.points().size())));
  REQUIRE(tier_sphere_error(result, 0) < 1e-5);
  REQUIRE(result.seam_max_angle[0] < 0.05);
  const auto filled = tf::make_filled_mesh(band.polygons(), result);
  REQUIRE(tf::is_manifold(filled.polygons()));
}

TEST_CASE("fill: a rim the table spans only by a protection split refuses "
          "where the rim may not be split",
          "[fill][chart]") {
  std::vector<tf::point<tier_real, 3>> loop{{-10.0, 0.0, 0.0},
                                            {10.0, 0.0, 0.0}};
  for (int k = 0; k < 6; ++k) {
    const double angle = 6.283185307179586 * double(k) / 6.0;
    loop.push_back({0.0, std::cos(angle), std::sin(angle)});
  }
  const auto mesh = hole_cone_mesh<tier_index, tier_real>(
      loop, tf::point<tier_real, 3>{0.0, 0.0, -20.0});
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  REQUIRE(rims.size() == 1);

  tf::hole_fill_config config;
  const auto split = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(split.status[0] == tf::hole_fill_status::filled);
  REQUIRE(split.charted[0] == tf::hole_chart_status::folded);
  REQUIRE(split.splits.size() > 0);
  const auto filled = tf::make_filled_mesh(mesh.polygons(), split);
  REQUIRE(tf::is_manifold(filled.polygons()));

  config.split_rim = false;
  const auto strict = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(strict.status[0] == tf::hole_fill_status::refused_rim_split);
  REQUIRE(strict.offending[0][0] == 0);
  REQUIRE(strict.offending[0][1] == 0);
  REQUIRE(strict.triangles[0].size() == 0);
  REQUIRE(strict.splits.size() == 0);
}

namespace {

/// The 24-vertex band of a 50-degree spherical cap, scaled to `span` mesh
/// lattice units across its rim, at the origin of a unit mesh; the cap's
/// rim is the boundary rim holding vertex 0.
auto tier_small_cap(double span) -> tf::polygons_buffer<tier_index, tier_real, 3, 3> {
  const auto band = hole_sphere_band_mesh<tier_index, tier_real>(24, 50.0, 4);
  const auto lattice = tf::exact::make_pt_converter<std::int64_t, tier_real>(
      tf::make_points(std::vector<tf::point<tier_real, 3>>{
          {0.5, 0.5, 0.5}, {-0.5, -0.5, -0.5}}));
  const double scale = span / (2.0 * std::sin(50.0 * 3.14159265358979 / 180.0) *
                               lattice.scale);
  tf::polygons_buffer<tier_index, tier_real, 3, 3> mesh;
  const std::size_t n = band.points().size();
  mesh.points_buffer().allocate(n + 3);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t axis = 0; axis < 3; ++axis)
      mesh.points()[i][axis] = band.points()[i][axis] * scale;
  mesh.points()[n] = tf::point<tier_real, 3>{0.5, 0.5, 0.5};
  mesh.points()[n + 1] = tf::point<tier_real, 3>{-0.5, 0.5, -0.5};
  mesh.points()[n + 2] = tf::point<tier_real, 3>{0.5, -0.5, -0.5};
  mesh.faces_buffer().allocate(band.faces().size() + 1);
  for (std::size_t f = 0; f < band.faces().size(); ++f)
    for (std::size_t c = 0; c < 3; ++c)
      mesh.faces()[f][c] = band.faces()[f][c];
  const std::size_t last = band.faces().size();
  mesh.faces()[last][0] = tier_index(n);
  mesh.faces()[last][1] = tier_index(n + 1);
  mesh.faces()[last][2] = tier_index(n + 2);
  return mesh;
}

} // namespace

TEST_CASE("fill: a chart stands on the mesh lattice down to its resolution",
          "[fill][chart]") {
  const auto run = [](double span) {
    const auto mesh = tier_small_cap(span);
    const auto rims = hole_rim_holding(tf::make_boundary_rims(mesh.polygons()),
                                       tier_index(0));
    REQUIRE(rims.size() == 1);
    const auto result = tf::fill_holes(mesh.polygons(), rims);
    REQUIRE(result.status[0] == tf::hole_fill_status::filled);
    double radius = 0.0, edge = 0.0;
    const auto loop = rims.vertices[0];
    for (std::size_t k = 0; k < std::size_t(loop.size()); ++k) {
      const auto p = mesh.points()[loop[k]];
      const auto q = mesh.points()[loop[(k + 1) % loop.size()]];
      radius = std::max(radius, std::sqrt(p[0] * p[0] + p[1] * p[1] +
                                          p[2] * p[2]));
      edge += std::sqrt((p[0] - q[0]) * (p[0] - q[0]) +
                        (p[1] - q[1]) * (p[1] - q[1]) +
                        (p[2] - q[2]) * (p[2] - q[2]));
    }
    edge /= double(loop.size());
    double worst = 0.0;
    for (const auto &p : result.minted_points[0])
      worst = std::max(worst, std::abs(std::sqrt(p[0] * p[0] + p[1] * p[1] +
                                                 p[2] * p[2]) -
                                       radius));
    return std::make_pair(result.charted[0], worst / edge);
  };
  // Far above the resolution the chart continues the sphere as well as a
  // chart of its own range; near it the error grows; at a few units per rim
  // edge the continuation is refused and the planar rim is lidded.
  for (double span : {1e6, 4096.0}) {
    const auto [charted, error] = run(span);
    REQUIRE(charted == tf::hole_chart_status::admitted);
    REQUIRE(error < 0.025);
  }
  {
    const auto [charted, error] = run(64.0);
    REQUIRE(charted == tf::hole_chart_status::admitted);
    REQUIRE(error < 0.5);
  }
  REQUIRE(run(16.0).first == tf::hole_chart_status::rim_plane);
}
