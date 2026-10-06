/**
 * @file test_hole_preflight.cpp
 * @brief Tests for the order preflight validates a rim in
 *
 * Tests for:
 * - tf::fill::preflight_hole_rim
 * - tf::fill::reject_overlapping_hole_rims
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/fill/fill_holes.hpp>
#include <trueform/fill/holes/preflight_hole_rim.hpp>
#include <trueform/fill/holes/reject_overlapping_hole_rims.hpp>
#include <trueform/fill/holes/validate_hole_rim.hpp>
#include <trueform/topology/face_membership.hpp>
#include "fill_generators.hpp"
#include <array>
#include <cstddef>
#include <vector>

namespace {

using preflight_index = int;
using preflight_int = tf::exact::int32;
using preflight_real = float;

enum class preflight_carrier { triangle, quad, flat };

/**
 * @brief A square rim (vertices 0..3) whose four edges are carried by four
 *        faces, each stated as asked: a triangle, a face of four corners, or
 *        a triangle whose own corners are collinear.
 */
auto preflight_carrier_mesh(preflight_carrier first, preflight_carrier third)
    -> tf::polygons_buffer<preflight_index, preflight_real, 3,
                           tf::dynamic_size> {
  tf::polygons_buffer<preflight_index, preflight_real, 3, tf::dynamic_size>
      mesh;
  mesh.points_buffer().emplace_back(0.0f, 0.0f, 0.0f);
  mesh.points_buffer().emplace_back(4.0f, 0.0f, 0.0f);
  mesh.points_buffer().emplace_back(4.0f, 4.0f, 0.0f);
  mesh.points_buffer().emplace_back(0.0f, 4.0f, 0.0f);
  mesh.points_buffer().emplace_back(2.0f, 2.0f, -4.0f);
  mesh.points_buffer().emplace_back(2.0f, -2.0f, -4.0f);
  mesh.points_buffer().emplace_back(8.0f, 0.0f, 0.0f);
  mesh.points_buffer().emplace_back(-4.0f, 4.0f, 0.0f);

  const auto carry = [&mesh](preflight_index a, preflight_index b,
                             preflight_index collinear,
                             preflight_carrier kind) {
    std::vector<preflight_index> corners{a, b};
    if (kind == preflight_carrier::flat)
      corners.push_back(collinear);
    else if (kind == preflight_carrier::quad) {
      corners.push_back(preflight_index(4));
      corners.push_back(preflight_index(5));
    } else
      corners.push_back(preflight_index(4));
    mesh.faces_buffer().push_back(tf::make_range(corners));
  };
  carry(0, 1, 6, first);
  carry(1, 2, 4, preflight_carrier::triangle);
  carry(2, 3, 7, third);
  carry(3, 0, 4, preflight_carrier::triangle);
  return mesh;
}

auto preflight_status(
    const tf::polygons_buffer<preflight_index, preflight_real, 3,
                              tf::dynamic_size> &mesh)
    -> tf::fill::hole_preflight_verdict<preflight_index> {
  const auto rims = hole_single_rim<preflight_index>({0, 1, 2, 3},
                                                     {0, 1, 2, 3});
  tf::face_membership<preflight_index> membership;
  membership.build(mesh.polygons());
  const auto converter =
      tf::exact::make_pt_converter<preflight_int, preflight_real>(
          mesh.points());
  tf::fill::hole_rim_scratch<preflight_index, preflight_int> scratch;
  tf::fill::hole_rim<preflight_index, preflight_int, preflight_real> rim;
  return tf::fill::preflight_hole_rim(mesh.polygons(), membership, rims, 0,
                                      converter, scratch, rim);
}

} // namespace

TEST_CASE("fill: a structural refusal outranks an unusable carrier whichever "
          "ticket is smaller",
          "[fill][preflight]") {
  const auto quad_first = preflight_status(preflight_carrier_mesh(
      preflight_carrier::quad, preflight_carrier::flat));
  REQUIRE(quad_first.status == tf::hole_fill_status::refused_invalid_rim);
  REQUIRE(quad_first.edge == 0);

  const auto flat_first = preflight_status(preflight_carrier_mesh(
      preflight_carrier::flat, preflight_carrier::quad));
  REQUIRE(flat_first.status == tf::hole_fill_status::refused_invalid_rim);
  REQUIRE(flat_first.edge == 2);
}

TEST_CASE("fill: unusable carriers report their lowest ticket",
          "[fill][preflight]") {
  const auto verdict = preflight_status(preflight_carrier_mesh(
      preflight_carrier::flat, preflight_carrier::flat));
  REQUIRE(verdict.status == tf::hole_fill_status::refused_host_face);
  REQUIRE(verdict.edge == 0);
}

TEST_CASE("fill: rims that claim one edge all refuse", "[fill][preflight]") {
  tf::boundary_rims<preflight_index> rims;
  rims.vertices.offsets_buffer().push_back(0);
  for (preflight_index v : {0, 1, 2})
    rims.vertices.data_buffer().push_back(v);
  rims.vertices.offsets_buffer().push_back(3);
  for (preflight_index v : {2, 1, 3})
    rims.vertices.data_buffer().push_back(v);
  rims.vertices.offsets_buffer().push_back(6);
  rims.faces.offsets_buffer().push_back(0);
  for (preflight_index f : {0, 1, 2})
    rims.faces.data_buffer().push_back(f);
  rims.faces.offsets_buffer().push_back(3);
  for (preflight_index f : {3, 4, 5})
    rims.faces.data_buffer().push_back(f);
  rims.faces.offsets_buffer().push_back(6);
  rims.closed.push_back(true);
  rims.closed.push_back(true);

  tf::buffer<preflight_index> rejected;
  tf::fill::reject_overlapping_hole_rims(rims, rejected);
  REQUIRE(rejected.size() == 2);
  REQUIRE(rejected[0] == 1);
  REQUIRE(rejected[1] == 0);
}

TEST_CASE("fill: the rim's rooting and direction decide the patch, not the "
          "order it was walked in",
          "[fill][preflight]") {
  std::vector<tf::point<preflight_real, 3>> loop{
      {1.0f, 0.0f, 0.3f},  {0.5f, 0.9f, -0.2f}, {-0.6f, 0.8f, 0.4f},
      {-1.0f, 0.1f, 0.0f}, {-0.4f, -0.9f, 0.5f}, {0.7f, -0.7f, -0.3f}};
  const auto mesh = hole_cone_mesh<preflight_index, preflight_real>(
      loop, tf::point<preflight_real, 3>{0.0f, 0.0f, -4.0f});
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  REQUIRE(rims.size() == 1);
  const auto reference = tf::fill_holes(mesh.polygons(), rims);
  REQUIRE(reference.status[0] == tf::hole_fill_status::filled);

  for (preflight_index root = 0; root < 6; ++root)
    for (bool reverse : {false, true}) {
      const auto turned = hole_turned_rim(rims, preflight_index(0), root,
                                          reverse);
      const auto result = tf::fill_holes(mesh.polygons(), turned);
      REQUIRE(result.status[0] == tf::hole_fill_status::filled);
      REQUIRE(result.triangles[0].size() == reference.triangles[0].size());
      for (std::size_t k = 0; k < result.triangles[0].size(); ++k)
        REQUIRE(result.triangles[0][k] == reference.triangles[0][k]);
    }
}
