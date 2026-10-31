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
#include "carriers.hpp"
#include "nd_array.hpp"

#include "trueform/cpp/geometry/dirichlet_field.hpp"
#include "trueform/cpp/geometry/geodesic_distances.hpp"
#include "trueform/cpp/geometry/make_plane_mesh.hpp"
#include "trueform/cpp/topology/label_connected_components.hpp"
#include "trueform/cpp/topology/neighborhoods.hpp"
#include "trueform/cpp/topology/vertex_link.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using fields_index = tf::cpp::default_index_t;

template <typename Real>
using fields_owned = tf::cpp::test::owned_mesh<fields_index, Real>;

template <typename Real> auto fields_plane() -> fields_owned<Real> {
  return {tf::cpp::make_plane_mesh<fields_index>(Real{4}, Real{4}, 8, 8)};
}

template <typename Real>
auto fields_point(const fields_owned<Real> &owned, std::size_t k)
    -> std::array<double, 3> {
  const auto &data = owned.polygons.points_buffer().data_buffer();
  return {double(data[3 * k]), double(data[3 * k + 1]),
          double(data[3 * k + 2])};
}

template <typename Real>
auto fields_find(const fields_owned<Real> &owned, double x, double y)
    -> fields_index {
  for (std::size_t k = 0; k < owned.polygons.points_buffer().size(); ++k) {
    const auto p = fields_point(owned, k);
    if (p[0] == x && p[1] == y)
      return fields_index(k);
  }
  return fields_index(-1);
}

} // namespace

TEMPLATE_TEST_CASE("geodesic distances on a plane are the straight line",
                   "[cpp][geometry][geodesic]", float, double) {
  auto owned = fields_plane<TestType>();
  const auto mesh = owned.mesh();
  const auto seed = fields_find(owned, -2.0, -2.0);
  REQUIRE(seed >= 0);
  const auto seeds = tf::cpp::test::make_nd_array<fields_index>({seed}, {1});
  const auto distances = tf::cpp::geodesic_distances(mesh, seeds);
  REQUIRE(distances.ndim() == 1);
  REQUIRE(distances.length() == mesh.number_of_points());
  const auto source = fields_point(owned, std::size_t(seed));
  for (std::size_t k = 0; k < distances.length(); ++k) {
    const auto p = fields_point(owned, k);
    const double line = std::hypot(p[0] - source[0], p[1] - source[1]);
    CHECK(std::abs(distances[k] - line) <= 1e-6 * (1.0 + line));
  }

  const auto reached = tf::cpp::geodesic_distances(mesh, seeds, 2.0);
  for (std::size_t k = 0; k < reached.length(); ++k)
    CHECK((reached[k] <= 2.0 || std::isinf(reached[k])));

  const auto none = tf::cpp::geodesic_distances(
      mesh, tf::cpp::test::make_nd_array<fields_index>({}, {0}));
  for (const auto distance : none)
    CHECK(std::isinf(distance));

  CHECK_THROWS_AS(
      tf::cpp::geodesic_distances(
          mesh, tf::cpp::test::make_nd_array<fields_index>({81}, {1})),
      std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::geodesic_distances(
                      mesh, seeds, std::numeric_limits<double>::quiet_NaN()),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::geodesic_distances(mesh, seeds, -1.0),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE("a cotangent Dirichlet field on a plane is linear",
                   "[cpp][geometry][dirichlet]", float, double) {
  auto owned = fields_plane<TestType>();
  const auto mesh = owned.mesh();
  std::vector<fields_index> ids;
  std::vector<double> values;
  for (std::size_t k = 0; k < mesh.number_of_points(); ++k) {
    const auto p = fields_point(owned, k);
    if (std::abs(p[0]) == 2.0 || std::abs(p[1]) == 2.0) {
      ids.push_back(fields_index(k));
      values.push_back(p[0] - 2.0 * p[1]);
    }
  }
  const auto fixed = tf::cpp::test::make_nd_array(ids, {int(ids.size())});
  const auto held = tf::cpp::test::make_nd_array(values, {int(values.size())});
  const auto result = tf::cpp::dirichlet_field(mesh, fixed, held,
                                               tf::dirichlet_order::harmonic);
  REQUIRE(result.status == tf::dirichlet_field_status::solved);
  REQUIRE(result.field.length() == mesh.number_of_points());
  for (std::size_t k = 0; k < mesh.number_of_points(); ++k) {
    const auto p = fields_point(owned, k);
    CHECK(std::abs(result.field[k] - (p[0] - 2.0 * p[1])) < 1e-6);
  }

  std::vector<double> uniform(3 * mesh.number_of_faces(), 1.0);
  const auto weights =
      tf::cpp::test::make_nd_array(uniform, {int(uniform.size())});
  const auto weighted = tf::cpp::dirichlet_field(
      mesh, weights, fixed, held, tf::dirichlet_order::biharmonic);
  CHECK(weighted.status == tf::dirichlet_field_status::solved);
  std::vector<double> negated(3 * mesh.number_of_faces(), -1.0);
  const auto indefinite =
      tf::cpp::test::make_nd_array(negated, {int(negated.size())});
  const auto refused = tf::cpp::dirichlet_field(mesh, indefinite, fixed, held,
                                                tf::dirichlet_order::harmonic);
  CHECK(refused.status == tf::dirichlet_field_status::refused_indefinite);
  for (std::size_t k = 0; k < ids.size(); ++k)
    CHECK(refused.field[std::size_t(ids[k])] == values[k]);

  const auto twice =
      tf::cpp::test::make_nd_array<fields_index>({ids[0], ids[0]}, {2});
  const auto pair = tf::cpp::test::make_nd_array<double>({0.0, 1.0}, {2});
  CHECK_THROWS_AS(tf::cpp::dirichlet_field(mesh, twice, pair,
                                           tf::dirichlet_order::harmonic),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::dirichlet_field(mesh, fixed, pair,
                                           tf::dirichlet_order::harmonic),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::dirichlet_field(mesh, held, fixed, held,
                                           tf::dirichlet_order::harmonic),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE("one seed's neighborhood is its block of all of them",
                   "[cpp][topology][neighborhoods]", float, double) {
  auto owned = fields_plane<TestType>();
  const auto mesh = owned.mesh();
  const auto all = tf::cpp::neighborhoods(mesh, TestType{1.1});
  for (fields_index seed = 0; seed < fields_index(mesh.number_of_points());
       seed += 7) {
    const auto one = tf::cpp::neighborhood(mesh, seed, TestType{1.1});
    const auto block = all.get(int(seed));
    std::vector<fields_index> left(one.begin(), one.end());
    std::vector<fields_index> right(block.begin(), block.end());
    std::sort(left.begin(), left.end());
    std::sort(right.begin(), right.end());
    CHECK(left == right);
    const auto with = tf::cpp::neighborhood(mesh, seed, TestType{1.1}, true);
    REQUIRE(with.length() == one.length() + 1);
    CHECK(with[0] == seed);
  }
  CHECK_THROWS_AS(tf::cpp::neighborhood(mesh, fields_index(81), TestType{1}),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::neighborhood(mesh, fields_index(0), TestType{0}),
                  std::invalid_argument);
}

TEST_CASE("a masked labelling walks only what the mask keeps",
          "[cpp][topology][components]") {
  const auto edges = tf::cpp::test::make_nd_array<fields_index>(
      {0, 1, 1, 2, 2, 3, 3, 4, 4, 5}, {5, 2});
  const auto link = tf::cpp::vertex_link_edges(edges, fields_index(6));
  const auto mask =
      tf::cpp::test::make_nd_array<std::int8_t>({1, 1, 0, 1, 1, 1}, {6});
  const auto labelled = tf::cpp::label_connected_components(link, mask);
  CHECK(labelled.n_components == 2);
  const std::vector<fields_index> expected{0, 0, -1, 1, 1, 1};
  for (std::size_t k = 0; k < 6; ++k)
    CHECK(labelled.labels[k] == expected[k]);
  const auto whole = tf::cpp::label_connected_components(link);
  CHECK(whole.n_components == 1);
  CHECK_THROWS_AS(
      tf::cpp::label_connected_components(
          link, tf::cpp::test::make_nd_array<std::int8_t>({1, 1}, {2})),
      std::invalid_argument);
}
