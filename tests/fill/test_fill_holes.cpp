/**
 * @file test_fill_holes.cpp
 * @brief Tests for the entry and the mesh it states
 *
 * Tests for:
 * - tf::fill_holes
 * - tf::make_filled_mesh
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/fill/fill_holes.hpp>
#include <trueform/fill/make_filled_mesh.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/topology/boundary_rims.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include "fill_generators.hpp"
#include "oneapi/tbb/global_control.h"
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using entry_index = int;
using entry_real = float;

auto entry_all_filled(const tf::hole_fill_result<entry_index, entry_real>
                          &result) -> bool {
  for (std::size_t group = 0; group < result.size(); ++group)
    if (result.status[group] != tf::hole_fill_status::filled)
      return false;
  return true;
}

auto entry_same(const tf::hole_fill_result<entry_index, entry_real> &a,
                const tf::hole_fill_result<entry_index, entry_real> &b)
    -> bool {
  if (a.size() != b.size() ||
      a.triangles.data_buffer().size() != b.triangles.data_buffer().size() ||
      a.minted_points.data_buffer().size() !=
          b.minted_points.data_buffer().size() ||
      a.splits.size() != b.splits.size())
    return false;
  for (std::size_t k = 0; k < a.triangles.data_buffer().size(); ++k)
    if (a.triangles.data_buffer()[k] != b.triangles.data_buffer()[k])
      return false;
  for (std::size_t k = 0; k < a.minted_points.data_buffer().size(); ++k)
    if (a.minted_points.data_buffer()[k] != b.minted_points.data_buffer()[k])
      return false;
  for (std::size_t k = 0; k < a.splits.size(); ++k)
    if (a.splits[k].face != b.splits[k].face ||
        a.splits[k].point != b.splits[k].point ||
        a.splits[k].parameter != b.splits[k].parameter)
      return false;
  for (std::size_t group = 0; group < a.size(); ++group)
    if (a.status[group] != b.status[group] ||
        a.refined[group] != b.refined[group] ||
        a.faired[group] != b.faired[group] ||
        a.seam_max_angle[group] != b.seam_max_angle[group])
      return false;
  return true;
}

/**
 * @brief The rhombus whose coarse patch the refinement splits once, so the
 *        fill states a point of its own strictly inside the patch.
 */
auto entry_rhombus_mesh()
    -> tf::polygons_buffer<entry_index, entry_real, 3, 3> {
  return hole_cone_mesh<entry_index, entry_real>(
      {{-4.0f, 0.0f, 0.0f},
       {0.0f, -1.0f, 0.0f},
       {4.0f, 0.0f, 0.0f},
       {0.0f, 1.0f, 0.0f}},
      tf::point<entry_real, 3>{0.0f, 0.0f, -4.0f});
}

/**
 * @brief A bowl whose hexagonal rim is carried by triangles alone, and whose
 *        rim vertex 0 also stands on a quad that touches the rim nowhere else.
 */
auto entry_vertex_ngon_mesh()
    -> tf::polygons_buffer<entry_index, entry_real, 3, tf::dynamic_size> {
  const std::vector<tf::point<entry_real, 3>> points{
      {2.0f, 0.0f, 0.3f},      {1.0f, 1.75f, -0.3f},  {-1.0f, 1.75f, 0.3f},
      {-2.0f, 0.0f, -0.3f},    {-1.0f, -1.75f, 0.3f}, {1.0f, -1.75f, -0.3f},
      {0.0f, 0.0f, -3.0f},     {3.0f, 1.5f, 0.5f},    {1.5f, 1.0f, -2.0f}};
  const std::vector<std::vector<entry_index>> faces{
      {1, 2, 6}, {2, 3, 6}, {3, 4, 6}, {4, 5, 6}, {5, 0, 6},
      {0, 1, 7}, {0, 7, 8, 6}, {8, 7, 1}, {6, 8, 1}};

  tf::polygons_buffer<entry_index, entry_real, 3, tf::dynamic_size> mesh;
  mesh.points_buffer().allocate(points.size());
  for (std::size_t k = 0; k < points.size(); ++k)
    mesh.points()[k] = points[k];
  auto &offsets = mesh.faces_buffer().offsets_buffer();
  auto &corners = mesh.faces_buffer().data_buffer();
  offsets.push_back(0);
  for (const auto &face : faces) {
    for (auto corner : face)
      corners.push_back(corner);
    offsets.push_back(entry_index(corners.size()));
  }
  return mesh;
}

} // namespace

TEST_CASE("fill: a punched box comes back closed and of its own volume",
          "[fill][entry]") {
  const auto box = tf::make_box_mesh<entry_index>(2.0f, 2.0f, 2.0f);
  const auto reference = double(tf::signed_volume(box.polygons()));
  for (const std::vector<std::size_t> &drop :
       {std::vector<std::size_t>{0}, std::vector<std::size_t>{0, 1},
        std::vector<std::size_t>{0, 1, 2, 3}}) {
    const auto holed = hole_punched_mesh(box, drop);
    const auto rims = tf::make_boundary_rims(holed.polygons());
    const auto result = tf::fill_holes(holed.polygons(), rims);
    REQUIRE(entry_all_filled(result));
    const auto filled = tf::make_filled_mesh(holed.polygons(), result);
    REQUIRE(tf::is_closed(filled.polygons()));
    REQUIRE(tf::is_manifold(filled.polygons()));
    REQUIRE(std::abs(double(tf::signed_volume(filled.polygons())) -
                     reference) < 1e-4 * reference);
  }
}

TEST_CASE("fill: a punched sphere comes back closed and near its own volume",
          "[fill][entry]") {
  const auto sphere = tf::make_sphere_mesh<entry_index>(1.0f, 10, 14);
  const auto reference = double(tf::signed_volume(sphere.polygons()));
  const auto holed =
      hole_punched_mesh(sphere, {5, 6, 7, 20, 21, 22, 23, 40, 41});
  const auto rims = tf::make_boundary_rims(holed.polygons());
  REQUIRE(rims.size() >= 1);
  const auto result = tf::fill_holes(holed.polygons(), rims);
  REQUIRE(entry_all_filled(result));
  const auto filled = tf::make_filled_mesh(holed.polygons(), result);
  REQUIRE(tf::is_closed(filled.polygons()));
  REQUIRE(tf::is_manifold(filled.polygons()));
  REQUIRE(std::abs(double(tf::signed_volume(filled.polygons())) - reference) <
          0.05 * reference);
}

TEST_CASE("fill: the product does not depend on how many workers ran it",
          "[fill][entry]") {
  const auto sphere = tf::make_sphere_mesh<entry_index>(1.0f, 8, 12);
  const auto holed = hole_punched_mesh(sphere, {3, 4, 15, 16, 30, 31, 32});
  const auto rims = tf::make_boundary_rims(holed.polygons());
  const auto reference = tf::fill_holes(holed.polygons(), rims);
  REQUIRE(entry_same(reference, tf::fill_holes(holed.polygons(), rims)));
  for (std::size_t workers : {std::size_t(1), std::size_t(2),
                              std::size_t(4)}) {
    const tbb::global_control control(
        tbb::global_control::max_allowed_parallelism, workers);
    REQUIRE(entry_same(reference, tf::fill_holes(holed.polygons(), rims)));
  }
}

TEST_CASE("fill: a planar rim the split scale can state is filled in its own "
          "plane",
          "[fill][entry]") {
  const auto plane = tf::make_plane_mesh<entry_index>(4.0f, 4.0f, 4, 4);
  const auto holed = hole_punched_mesh(plane, {10, 11, 12, 13});
  const auto rims = tf::make_boundary_rims(holed.polygons());
  REQUIRE(rims.size() == 2);
  const auto result = tf::fill_holes(holed.polygons(), rims);
  const std::size_t hole = rims.vertices[0].size() < rims.vertices[1].size()
                               ? 0
                               : 1;
  REQUIRE(result.status[hole] == tf::hole_fill_status::filled);
  REQUIRE(result.refined[hole] != tf::hole_refine_status::not_attempted);
  REQUIRE(result.triangles[hole].size() >= rims.vertices[hole].size() - 2);
}

TEST_CASE("fill: a planar rim with arbitrary coordinates is filled in its "
          "own plane",
          "[fill][entry]") {
  const std::vector<tf::point<entry_real, 3>> loop{
      {1.0f, 0.0f, 0.0f},      {0.53f, 0.81f, 0.0f},  {-0.47f, 0.77f, 0.0f},
      {-0.91f, 0.11f, 0.0f},   {-0.4f, -0.83f, 0.0f}, {0.61f, -0.72f, 0.0f}};
  const auto mesh = hole_cone_mesh<entry_index, entry_real>(
      loop, tf::point<entry_real, 3>{0.0f, 0.0f, -4.0f});
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  REQUIRE(rims.size() == 1);
  const auto result = tf::fill_holes(mesh.polygons(), rims);
  REQUIRE(result.status[0] == tf::hole_fill_status::filled);
  REQUIRE(result.refined[0] != tf::hole_refine_status::not_attempted);
  REQUIRE(result.triangles[0].size() >= rims.vertices[0].size() - 2);
  const auto filled = tf::make_filled_mesh(mesh.polygons(), result);
  REQUIRE(tf::is_closed(filled.polygons()));
}

TEST_CASE("fill: fairing moves what the refinement minted and leaves the "
          "patch it was handed",
          "[fill][entry]") {
  const auto mesh = entry_rhombus_mesh();
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  REQUIRE(rims.size() == 1);
  tf::hole_fill_config config;
  config.min_quality = 0.25;

  config.fairing = false;
  const auto loose = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(loose.status[0] == tf::hole_fill_status::filled);
  REQUIRE(loose.refined[0] == tf::hole_refine_status::floor_met);
  REQUIRE(loose.faired[0] == tf::hole_fair_status::fairing_off);
  REQUIRE(loose.minted_points[0].size() == 1);
  REQUIRE(loose.triangles[0].size() == 4);
  REQUIRE(loose.seam_max_angle[0] > 0.0f);

  config.fairing = true;
  const auto faired = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(faired.faired[0] == tf::hole_fair_status::faired);
  REQUIRE(faired.refined[0] == loose.refined[0]);
  REQUIRE(faired.triangles.data_buffer().size() ==
          loose.triangles.data_buffer().size());
  for (std::size_t k = 0; k < loose.triangles.data_buffer().size(); ++k)
    REQUIRE(faired.triangles.data_buffer()[k] ==
            loose.triangles.data_buffer()[k]);
  REQUIRE(faired.minted_points[0][0][2] != loose.minted_points[0][0][2]);
  REQUIRE(faired.seam_max_angle[0] <= loose.seam_max_angle[0]);

  const auto filled = tf::make_filled_mesh(mesh.polygons(), faired);
  REQUIRE(tf::is_closed(filled.polygons()));
  REQUIRE(tf::is_manifold(filled.polygons()));
}

TEST_CASE("fill: a rim vertex standing on an n-gon refuses the fairing alone",
          "[fill][entry]") {
  const auto mesh = entry_vertex_ngon_mesh();
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  REQUIRE(rims.size() == 1);
  REQUIRE(rims.vertices[0].size() == 6);
  tf::hole_fill_config config;

  config.fairing = false;
  const auto loose = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(loose.status[0] == tf::hole_fill_status::filled);
  REQUIRE(loose.faired[0] == tf::hole_fair_status::fairing_off);

  config.fairing = true;
  const auto asked = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(asked.status[0] == tf::hole_fill_status::filled);
  REQUIRE(asked.faired[0] == tf::hole_fair_status::fairing_refused);
  REQUIRE(asked.seam_max_angle[0] >= 0.0f);
  REQUIRE(asked.triangles.data_buffer().size() ==
          loose.triangles.data_buffer().size());
  for (std::size_t k = 0; k < loose.triangles.data_buffer().size(); ++k)
    REQUIRE(asked.triangles.data_buffer()[k] ==
            loose.triangles.data_buffer()[k]);
  REQUIRE(asked.minted_points.data_buffer().size() ==
          loose.minted_points.data_buffer().size());
  for (std::size_t k = 0; k < loose.minted_points.data_buffer().size(); ++k)
    for (std::size_t axis = 0; axis < 3; ++axis)
      REQUIRE(asked.minted_points.data_buffer()[k][axis] ==
              loose.minted_points.data_buffer()[k][axis]);
}

TEST_CASE("fill: a faired product does not depend on how many workers ran it",
          "[fill][entry]") {
  const auto mesh = entry_rhombus_mesh();
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  tf::hole_fill_config config;
  config.min_quality = 0.25;
  const auto reference = tf::fill_holes(mesh.polygons(), rims, config);
  REQUIRE(reference.faired[0] == tf::hole_fair_status::faired);
  for (std::size_t workers : {std::size_t(1), std::size_t(2),
                              std::size_t(4)}) {
    const tbb::global_control control(
        tbb::global_control::max_allowed_parallelism, workers);
    REQUIRE(entry_same(reference,
                       tf::fill_holes(mesh.polygons(), rims, config)));
  }
}

TEST_CASE("fill: a mesh of mixed arity stitches at the arity it had",
          "[fill][entry]") {
  const auto mesh = entry_vertex_ngon_mesh();
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  REQUIRE(rims.size() == 1);
  const auto result = tf::fill_holes(mesh.polygons(), rims);
  REQUIRE(entry_all_filled(result));

  const auto filled = tf::make_filled_mesh(mesh.polygons(), result);
  REQUIRE(tf::is_closed(filled.polygons()));
  REQUIRE(tf::is_manifold(filled.polygons()));
  REQUIRE(filled.points().size() ==
          mesh.points().size() + result.minted_points.data_buffer().size());

  std::size_t planned = 0;
  for (std::size_t plan = 0; plan < result.plan_faces.size(); ++plan)
    planned += result.plan_triangles[plan].size();
  REQUIRE(filled.faces().size() ==
          mesh.faces().size() - result.plan_faces.size() + planned +
              result.triangles.data_buffer().size());

  std::size_t quads = 0;
  for (std::size_t face = 0; face < filled.faces().size(); ++face)
    quads += filled.faces()[face].size() == 4 ? 1 : 0;
  REQUIRE(quads == 1);
}

TEST_CASE("fill: a form tagged with its face membership fills alike",
          "[fill][entry]") {
  const auto sphere = tf::make_sphere_mesh<entry_index>(1.0f, 8, 12);
  const auto holed = hole_punched_mesh(sphere, {3, 4, 15, 16, 30, 31, 32});
  const auto rims = tf::make_boundary_rims(holed.polygons());
  tf::face_membership<entry_index> membership;
  membership.build(holed.polygons());

  const auto bare = tf::fill_holes(holed.polygons(), rims);
  REQUIRE(entry_all_filled(bare));
  REQUIRE(entry_same(
      bare, tf::fill_holes(holed.polygons() | tf::tag(membership), rims)));
}

TEST_CASE("fill: a refused group publishes nothing", "[fill][entry]") {
  const auto plane = tf::make_plane_mesh<entry_index>(4.0f, 4.0f, 3, 3);
  const auto rims = tf::make_boundary_rims(plane.polygons());
  const auto result = tf::fill_holes(plane.polygons(), rims);
  REQUIRE(result.size() == 1);
  REQUIRE(result.status[0] != tf::hole_fill_status::filled);
  REQUIRE(result.triangles[0].size() == 0);
  REQUIRE(result.minted_points[0].size() == 0);
  REQUIRE(result.splits.size() == 0);
  REQUIRE(result.plan_faces.size() == 0);
  REQUIRE(result.refined[0] == tf::hole_refine_status::not_attempted);
  REQUIRE(result.faired[0] == tf::hole_fair_status::not_attempted);
  const auto filled = tf::make_filled_mesh(plane.polygons(), result);
  REQUIRE(filled.faces().size() == plane.faces().size());
  REQUIRE(filled.points().size() == plane.points().size());
}
