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
#include "async_resolver.hpp"
#include "carriers.hpp"

#include "trueform/core/buffer.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/fill/async/fill_holes.hpp"
#include "trueform/cpp/fill/async/filled_mesh.hpp"
#include "trueform/cpp/fill/fill_holes.hpp"
#include "trueform/cpp/fill/filled_mesh.hpp"
#include "trueform/cpp/fill/hole_fill_report.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/topology/boundary_rims.hpp"
#include "trueform/cpp/topology/is_closed.hpp"
#include "trueform/cpp/topology/is_manifold.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using fill_index = tf::cpp::default_index_t;

template <typename Real>
using fill_owned_t = tf::cpp::test::owned_mesh<fill_index, Real>;

constexpr auto fill_filled =
    static_cast<std::int8_t>(tf::hole_fill_status::filled);

/// A box of three ticks a side with the named faces dropped. Faces 8 and 9
/// are the centre cell of its +z side and 26 and 27 that of its -z side, so
/// dropping all four opens two rims that share no vertex.
template <typename Real>
auto fill_punched_box(std::initializer_list<std::size_t> drop)
    -> fill_owned_t<Real> {
  const auto box = tf::cpp::make_box_mesh(Real{2}, Real{2}, Real{2}, 3, 3, 3);
  const auto &corners = box.faces_buffer().data_buffer();
  std::vector<fill_index> indices;
  for (std::size_t face = 0; face < box.size(); ++face)
    if (std::find(drop.begin(), drop.end(), face) == drop.end())
      for (std::size_t k = 0; k < 3; ++k)
        indices.push_back(corners[3 * face + k]);
  fill_owned_t<Real> owned;
  owned.polygons = tf::cpp::test::polygons_of<fill_index, Real>(
      indices, box.points_buffer().data_buffer());
  return owned;
}

template <typename Real> auto fill_two_holed_box() -> fill_owned_t<Real> {
  return fill_punched_box<Real>({8, 9, 26, 27});
}

/// The cone over a rhombus, whose coarse patch the refinement splits.
template <typename Real> auto fill_rhombus_cone() -> fill_owned_t<Real> {
  fill_owned_t<Real> owned;
  owned.polygons = tf::cpp::test::polygons_of<fill_index, Real>(
      {0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4},
      {Real{-4}, Real{0}, Real{0}, Real{0}, Real{-1}, Real{0}, Real{4}, Real{0},
       Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{-4}});
  return owned;
}

/// Three triangles on one edge: every rim runs between its two ends, open.
template <typename Real> auto fill_book() -> fill_owned_t<Real> {
  fill_owned_t<Real> owned;
  owned.polygons = tf::cpp::test::polygons_of<fill_index, Real>(
      {0, 1, 2, 1, 0, 3, 0, 1, 4},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0.5}, Real{1},
       Real{0}, Real{0.5}, Real{-1}, Real{0}, Real{0.5}, Real{0}, Real{1}});
  return owned;
}

auto fill_ids(std::initializer_list<fill_index> ids)
    -> tf::cpp::nd_array<fill_index> {
  tf::buffer<fill_index> buffer;
  buffer.allocate(ids.size());
  std::copy(ids.begin(), ids.end(), buffer.begin());
  return tf::cpp::nd_array<fill_index>::from_buffer(
      std::move(buffer), {static_cast<int>(ids.size())});
}

template <typename T>
auto fill_same_lane(const tf::cpp::nd_array<T> &a,
                    const tf::cpp::nd_array<T> &b) -> bool {
  return a.raw_shape() == b.raw_shape() &&
         std::equal(a.begin(), a.end(), b.begin(), b.end());
}

template <typename Real>
auto fill_same_report(const tf::cpp::hole_fill_report<fill_index, Real> &a,
                      const tf::cpp::hole_fill_report<fill_index, Real> &b)
    -> bool {
  return fill_same_lane(a.status, b.status) &&
         fill_same_lane(a.refined, b.refined) &&
         fill_same_lane(a.faired, b.faired) &&
         fill_same_lane(a.triangle_offsets, b.triangle_offsets) &&
         fill_same_lane(a.triangles, b.triangles) &&
         fill_same_lane(a.minted_points, b.minted_points) &&
         fill_same_lane(a.splits, b.splits) &&
         fill_same_lane(a.plan_triangles, b.plan_triangles);
}

auto fill_rim_vertices(const tf::cpp::boundary_rims_result<fill_index> &rims,
                       int rim) -> std::vector<fill_index> {
  const auto vertices = rims.vertices.get(rim);
  std::vector<fill_index> sorted(vertices.begin(), vertices.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

} // namespace

TEMPLATE_TEST_CASE("fill_holes fills every rim and filled_mesh closes the mesh",
                   "[cpp][fill][sync]", float, double) {
  const auto source = fill_two_holed_box<TestType>();
  const auto rims = tf::cpp::boundary_rims(source.mesh());
  REQUIRE(rims.closed.length() == 2);
  REQUIRE(rims.closed[0] != 0);
  REQUIRE(rims.closed[1] != 0);

  const auto report = tf::cpp::fill_holes(source.mesh());
  static_assert(
      std::is_same_v<decltype(report),
                     const tf::cpp::hole_fill_report<fill_index, TestType>>);
  REQUIRE(static_cast<bool>(report.result));
  REQUIRE(report.status.length() == 2);
  REQUIRE(report.offending.shape_at(1) == 2);
  for (std::size_t group = 0; group < 2; ++group) {
    CHECK(report.status[group] == fill_filled);
    CHECK(report.offending[2 * group] == -1);
    CHECK(report.triangle_offsets[group + 1] > report.triangle_offsets[group]);
  }
  REQUIRE(report.triangles.shape_at(1) == 3);
  REQUIRE(report.triangles.shape_at(0) == report.triangle_offsets[2]);
  REQUIRE(report.minted_points.shape_at(1) == 3);
  REQUIRE(report.minted_points.shape_at(0) == report.minted_point_offsets[2]);

  fill_owned_t<TestType> filled;
  filled.polygons = tf::cpp::filled_mesh(source.mesh(), report);
  CHECK(tf::cpp::is_closed(filled.mesh()));
  CHECK(tf::cpp::is_manifold(filled.mesh()));
  CHECK(filled.mesh().number_of_points() ==
        source.mesh().number_of_points() +
            static_cast<std::size_t>(report.minted_points.shape_at(0)));
  CHECK(filled.mesh().number_of_faces() ==
        source.mesh().number_of_faces() - report.plan_faces.length() +
            static_cast<std::size_t>(report.plan_triangles.shape_at(0) +
                                     report.triangles.shape_at(0)));
}

TEMPLATE_TEST_CASE("fill_holes with rim ids fills the named rims alone",
                   "[cpp][fill][sync]", float, double) {
  const auto source = fill_two_holed_box<TestType>();
  const auto rims = tf::cpp::boundary_rims(source.mesh());
  REQUIRE(rims.closed.length() == 2);

  const auto report = tf::cpp::fill_holes(source.mesh(), fill_ids({1}));
  REQUIRE(report.status.length() == 1);
  CHECK(report.status[0] == fill_filled);

  fill_owned_t<TestType> filled;
  filled.polygons = tf::cpp::filled_mesh(source.mesh(), report);
  CHECK_FALSE(tf::cpp::is_closed(filled.mesh()));
  const auto left = tf::cpp::boundary_rims(filled.mesh());
  REQUIRE(left.closed.length() == 1);
  CHECK(fill_rim_vertices(left, 0) == fill_rim_vertices(rims, 0));
}

TEMPLATE_TEST_CASE("fill_holes and filled_mesh refuse what they cannot state",
                   "[cpp][fill][sync]", float, double) {
  const auto source = fill_two_holed_box<TestType>();
  CHECK_THROWS_AS(tf::cpp::fill_holes(source.mesh(), fill_ids({2})),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::fill_holes(source.mesh(), fill_ids({-1})),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::fill_holes(source.mesh(), fill_ids({0, 0})),
                  std::invalid_argument);
  tf::hole_fill_config negative;
  negative.min_quality = -1;
  CHECK_THROWS_AS(tf::cpp::fill_holes(source.mesh(), negative),
                  std::invalid_argument);

  const auto book = fill_book<TestType>();
  const auto rims = tf::cpp::boundary_rims(book.mesh());
  REQUIRE(rims.closed.length() > 0);
  REQUIRE(rims.closed[0] == 0);
  CHECK_THROWS_AS(tf::cpp::fill_holes(book.mesh(), fill_ids({0})),
                  std::invalid_argument);

  const auto report = tf::cpp::fill_holes(book.mesh());
  REQUIRE(report.status.length() == rims.closed.length());
  for (std::size_t group = 0; group < report.status.length(); ++group) {
    CHECK(report.status[group] != fill_filled);
    CHECK(report.triangle_offsets[group + 1] == report.triangle_offsets[group]);
  }

  CHECK_THROWS_AS(tf::cpp::filled_mesh(source.mesh(), report),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::filled_mesh(source.mesh(),
                           tf::cpp::hole_fill_report<fill_index, TestType>{}),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE("fill_holes at min quality zero keeps the coarse patch",
                   "[cpp][fill][sync]", float, double) {
  const auto source = fill_rhombus_cone<TestType>();

  tf::hole_fill_config coarse_config;
  coarse_config.min_quality = 0;
  const auto coarse = tf::cpp::fill_holes(source.mesh(), coarse_config);
  REQUIRE(coarse.status.length() == 1);
  REQUIRE(coarse.status[0] == fill_filled);
  CHECK(coarse.triangles.shape_at(0) == 2);
  CHECK(coarse.minted_points.shape_at(0) == 0);

  const auto refined = tf::cpp::fill_holes(source.mesh());
  REQUIRE(refined.status[0] == fill_filled);
  CHECK(refined.minted_points.shape_at(0) > 0);

  fill_owned_t<TestType> filled;
  filled.polygons = tf::cpp::filled_mesh(source.mesh(), coarse);
  CHECK(tf::cpp::is_closed(filled.mesh()));
  CHECK(tf::cpp::is_manifold(filled.mesh()));
}

TEMPLATE_TEST_CASE("fill_holes and filled_mesh resolve as they do in place",
                   "[cpp][fill][async]", float, double) {
  const auto source = fill_two_holed_box<TestType>();
  const auto report = tf::cpp::fill_holes(source.mesh());
  const auto named = tf::cpp::fill_holes(source.mesh(), fill_ids({1, 0}));
  const auto filled = tf::cpp::filled_mesh(source.mesh(), report);

  CHECK(fill_same_report(tf::cpp::async::fill_holes(source.mesh()).get(),
                         report));

  tf::cpp::test::counting_resolver resolver;
  const auto resolved_named =
      tf::cpp::test::resolve_once(resolver, [&source](auto &submitter) {
        return tf::cpp::async::fill_holes(submitter, source.mesh(),
                                          fill_ids({1, 0}));
      });
  CHECK(fill_same_report(resolved_named, named));

  const auto resolved_filled = tf::cpp::test::resolve_once(
      resolver, [&source, &report](auto &submitter) {
        return tf::cpp::async::filled_mesh(submitter, source.mesh(), report);
      });
  const auto &faces = filled.faces_buffer().data_buffer();
  const auto &resolved_faces = resolved_filled.faces_buffer().data_buffer();
  CHECK(std::equal(faces.begin(), faces.end(), resolved_faces.begin(),
                   resolved_faces.end()));
  CHECK(resolved_filled.points_buffer().data_buffer().size() ==
        filled.points_buffer().data_buffer().size());
}
