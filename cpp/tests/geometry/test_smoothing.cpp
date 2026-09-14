/*
 * Copyright (c) 2025 XLAB
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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/geometry/async/laplacian_smoothed.hpp"
#include "trueform/cpp/geometry/async/taubin_smoothed.hpp"
#include "trueform/cpp/geometry/laplacian_smoothed.hpp"
#include "trueform/cpp/geometry/taubin_smoothed.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using smoothing_index = tf::cpp::default_index_t;

template <typename Real>
using smoothing_owned = tf::cpp::test::owned_mesh<smoothing_index, Real>;

template <typename Real>
using smoothing_result = tf::polygons_buffer<smoothing_index, Real, 3, 3>;

struct smoothing_counting_resolver {
  int *count;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    ++*count;
    return std::make_shared<state_type<T>>();
  }
};

template <typename Real> auto tetrahedron_mesh() -> smoothing_owned<Real> {
  return {tf::cpp::test::polygons_of<smoothing_index, Real>(
      {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3},
      {0, 0, 1, -1, -1, 0, 1, -1, 0, 0, 1, 0})};
}

template <typename Real> auto expanded_coordinates() -> std::array<Real, 24> {
  return {0, 0, 1, -1, -1, 0, 1, -1, 0, 0, 1, 0,
          4, 0, 1, 3,  -1, 0, 5, -1, 0, 4, 1, 0};
}

/// A test writes through its own storage and then says what moved, which is
/// the protocol the cache carries.
template <typename Real, typename Coordinates>
auto restate_points(smoothing_owned<Real> &owned, const Coordinates &values)
    -> void {
  auto &storage = owned.polygons.points_buffer().data_buffer();
  storage.allocate(values.size());
  std::copy(values.begin(), values.end(), storage.begin());
  owned.cache.points_changed();
}

template <typename Real>
auto coordinates_of(const smoothing_result<Real> &value)
    -> const tf::buffer<Real> & {
  return value.points_buffer().data_buffer();
}

} // namespace

TEMPLATE_TEST_CASE("async smoothing borrows its operand and matches the "
                   "synchronous call",
                   "[cpp][geometry][smoothing][async]", float, double) {
  auto input = tetrahedron_mesh<TestType>();
  const auto expected_laplacian =
      tf::cpp::laplacian_smoothed(input.mesh(), 2, TestType{0.25});
  const auto expected_taubin =
      tf::cpp::taubin_smoothed(input.mesh(), 2, TestType{0.4}, TestType{0.2});

  auto laplacian =
      tf::cpp::async::laplacian_smoothed(input.mesh(), 2, TestType{0.25});
  int submissions = 0;
  auto taubin = tf::cpp::async::taubin_smoothed(
      smoothing_counting_resolver{&submissions}, input.mesh(), 2, TestType{0.4},
      TestType{0.2});
  static_assert(std::is_same_v<decltype(laplacian),
                               std::future<smoothing_result<TestType>>>);
  static_assert(std::is_same_v<decltype(taubin),
                               std::future<smoothing_result<TestType>>>);
  CHECK(submissions == 1);

  // the job carries the mesh as it stands, so the caller keeps the arrays it
  // named alive until the future completes
  const auto actual_laplacian = laplacian.get();
  const auto actual_taubin = taubin.get();
  REQUIRE(coordinates_of(actual_laplacian).size() ==
          coordinates_of(expected_laplacian).size());
  REQUIRE(coordinates_of(actual_taubin).size() ==
          coordinates_of(expected_taubin).size());
  for (std::size_t index = 0; index < coordinates_of(actual_laplacian).size();
       ++index)
    CHECK(coordinates_of(actual_laplacian)[index] ==
          coordinates_of(expected_laplacian)[index]);
  for (std::size_t index = 0; index < coordinates_of(actual_taubin).size();
       ++index)
    CHECK(coordinates_of(actual_taubin)[index] ==
          coordinates_of(expected_taubin)[index]);

  auto failure = tf::cpp::async::laplacian_smoothed(input.mesh(), -1);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}

TEMPLATE_TEST_CASE("async smoothing fills the cache it was handed",
                   "[cpp][geometry][smoothing][async][cache]", float, double) {
  auto input = tetrahedron_mesh<TestType>();
  REQUIRE_FALSE(input.cache.is_vertex_link_built());

  // one job is one filler, which is the lazy use the contract allows; what it
  // fills is the caller's cache, because the mesh it carries is the caller's
  auto result = tf::cpp::async::taubin_smoothed(input.mesh(), 0);
  CHECK(result.get().points_buffer().size() ==
        input.polygons.points_buffer().size());
  CHECK(input.cache.is_vertex_link_built());
}

TEMPLATE_TEST_CASE("smoothing returns core's own storage at the input's arity",
                   "[cpp][geometry][smoothing]", float, double) {
  auto input = tetrahedron_mesh<TestType>();

  const auto laplacian = tf::cpp::laplacian_smoothed(input.mesh(), 1);
  const auto custom_laplacian =
      tf::cpp::laplacian_smoothed(input.mesh(), 1, TestType{0.25});
  constexpr auto lambda = TestType{0.4};
  constexpr auto kpb = TestType{0.2};
  const auto taubin = tf::cpp::taubin_smoothed(input.mesh(), 1, lambda, kpb);

  static_assert(
      std::is_same_v<decltype(laplacian), const smoothing_result<TestType>>);
  CHECK(laplacian.faces_buffer().size() == 4);
  CHECK(laplacian.points_buffer().size() == 4);
  CHECK(taubin.faces_buffer().size() == 4);
  CHECK(taubin.points_buffer().size() == 4);

  const auto tolerance = std::is_same_v<TestType, float> ? 1e-5 : 1e-12;
  CHECK(coordinates_of(laplacian)[2] == Catch::Approx(0.5).margin(tolerance));
  for (const auto index : {5U, 8U, 11U})
    CHECK(coordinates_of(laplacian)[index] ==
          Catch::Approx(1.0 / 6.0).margin(tolerance));
  CHECK(coordinates_of(custom_laplacian)[2] ==
        Catch::Approx(0.75).margin(tolerance));

  const auto mu = TestType{1} / (kpb - TestType{1} / lambda);
  const auto expected_taubin =
      (TestType{1} - lambda) +
      (lambda / TestType{3} - (TestType{1} - lambda)) * mu;
  CHECK(coordinates_of(taubin)[2] ==
        Catch::Approx(static_cast<double>(expected_taubin)).margin(tolerance));
}

TEMPLATE_TEST_CASE("smoothing reuses the source vertex-link cache",
                   "[cpp][geometry][smoothing][cache]", float, double) {
  auto input = tetrahedron_mesh<TestType>();
  REQUIRE_FALSE(input.cache.is_vertex_link_built());

  static_cast<void>(tf::cpp::laplacian_smoothed(input.mesh(), 1));
  REQUIRE(input.cache.is_vertex_link_fresh(input.mesh().geometry()));
  REQUIRE(input.cache.vertex_link_build_count() == 1);

  static_cast<void>(tf::cpp::taubin_smoothed(input.mesh(), 1));
  CHECK(input.cache.vertex_link_build_count() == 1);

  const auto extra_points = expanded_coordinates<TestType>();
  restate_points(input, extra_points);
  REQUIRE_FALSE(input.cache.is_vertex_link_fresh(input.mesh().geometry()));
  const auto expanded = tf::cpp::laplacian_smoothed(input.mesh(), 1);
  const auto expanded_reader = input.mesh();
  const auto expanded_link = expanded_reader.vertex_link();
  CHECK(input.cache.vertex_link_build_count() == 2);
  REQUIRE(expanded_link.size() == 8);
  for (int point = 0; point < 4; ++point) {
    const auto neighbors = expanded_link[point];
    REQUIRE(neighbors.size() == 3);
    std::array<bool, 4> seen{};
    for (const auto neighbor : neighbors) {
      REQUIRE(neighbor >= 0);
      REQUIRE(neighbor < 4);
      REQUIRE(neighbor != point);
      seen[static_cast<std::size_t>(neighbor)] = true;
    }
    for (int expected = 0; expected < 4; ++expected)
      CHECK(seen[static_cast<std::size_t>(expected)] == (expected != point));
  }
  for (int point = 4; point < 8; ++point)
    CHECK(expanded_link[point].size() == 0);
  REQUIRE(expanded.points_buffer().size() == 8);
  for (std::size_t index = 12; index < extra_points.size(); ++index)
    CHECK(coordinates_of(expanded)[index] == extra_points[index]);
}

TEMPLATE_TEST_CASE("smoothing states arrays of its own and carries no frame",
                   "[cpp][geometry][smoothing][ownership]", float, double) {
  auto input = tetrahedron_mesh<TestType>();
  input.place({TestType{1}, TestType{0}, TestType{0}, TestType{10}, TestType{0},
               TestType{1}, TestType{0}, TestType{20}, TestType{0}, TestType{0},
               TestType{1}, TestType{30}, TestType{0}, TestType{0}, TestType{0},
               TestType{1}});

  const auto output = tf::cpp::laplacian_smoothed(input.mesh(), 1);
  const auto first_face = output.faces_buffer().data_buffer()[0];
  const auto first_point = coordinates_of(output)[0];

  // the result is storage of its own; only the faces it states are the
  // input's, and a placement is the caller's to keep rather than the
  // result's to carry
  CHECK(coordinates_of(output).data() !=
        input.polygons.points_buffer().data_buffer().data());
  CHECK(output.faces_buffer().data_buffer().data() !=
        input.polygons.faces_buffer().data_buffer().data());
  CHECK(output.faces_buffer().data_buffer()[0] ==
        input.polygons.faces_buffer().data_buffer()[0]);
  CHECK(coordinates_of(output)[2] == Catch::Approx(0.5));

  input = {};
  CHECK(output.faces_buffer().data_buffer()[0] == first_face);
  CHECK(coordinates_of(output)[0] == first_point);
}

TEMPLATE_TEST_CASE("smoothing rejects negative iterations and unsafe faces",
                   "[cpp][geometry][smoothing][validation]", float, double) {
  // the empty carrier is a mesh, and an empty mesh smooths to itself
  const smoothing_owned<TestType> empty;
  CHECK(tf::cpp::laplacian_smoothed(empty.mesh(), 1).points_buffer().size() ==
        0);
  CHECK(tf::cpp::taubin_smoothed(empty.mesh(), 1).points_buffer().size() == 0);

  auto input = tetrahedron_mesh<TestType>();
  CHECK_THROWS_AS(tf::cpp::laplacian_smoothed(input.mesh(), -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::taubin_smoothed(input.mesh(), -1),
                  std::invalid_argument);
  CHECK_FALSE(input.cache.is_vertex_link_built());

  // the door is the cache's and it answers for the READING, so a corner that
  // names no point of it is refused however the storage came to say so
  smoothing_owned<TestType> negative;
  negative.polygons = tf::cpp::test::polygons_of<smoothing_index, TestType>(
      {0, 1, -1}, {0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK_THROWS_AS(tf::cpp::laplacian_smoothed(negative.mesh(), 1),
                  std::out_of_range);

  auto outrunning = tetrahedron_mesh<TestType>();
  restate_points(outrunning, std::array<TestType, 6>{0, 0, 0, 1, 0, 0});
  CHECK_THROWS_AS(tf::cpp::laplacian_smoothed(outrunning.mesh(), 1),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::taubin_smoothed(outrunning.mesh(), 1),
                  std::out_of_range);
}

TEMPLATE_TEST_CASE("smoothing preserves zero iterations",
                   "[cpp][geometry][smoothing]", float, double) {
  auto input = tetrahedron_mesh<TestType>();
  const auto &source = input.polygons.points_buffer().data_buffer();

  const auto laplacian = tf::cpp::laplacian_smoothed(input.mesh(), 0);
  const auto taubin = tf::cpp::taubin_smoothed(input.mesh(), 0);

  REQUIRE(input.cache.is_vertex_link_fresh(input.mesh().geometry()));
  REQUIRE(coordinates_of(laplacian).size() == source.size());
  REQUIRE(coordinates_of(taubin).size() == source.size());
  CHECK(coordinates_of(laplacian).data() != source.data());
  CHECK(coordinates_of(taubin).data() != source.data());
  for (std::size_t index = 0; index < source.size(); ++index) {
    CHECK(coordinates_of(laplacian)[index] == source[index]);
    CHECK(coordinates_of(taubin)[index] == source[index]);
  }
}

TEST_CASE("smoothing float and double APIs link from the native archive",
          "[cpp][geometry][smoothing][link]") {
  auto float_input = tetrahedron_mesh<float>();
  auto double_input = tetrahedron_mesh<double>();
  auto (*laplacian_float)(const tf::cpp::mesh<smoothing_index, float> &, int,
                          float)
      ->smoothing_result<float> =
      &tf::cpp::laplacian_smoothed<smoothing_index, float, 3, 3>;
  auto (*laplacian_double)(const tf::cpp::mesh<smoothing_index, double> &, int,
                           double)
      ->smoothing_result<double> =
      &tf::cpp::laplacian_smoothed<smoothing_index, double, 3, 3>;
  auto (*taubin_float)(const tf::cpp::mesh<smoothing_index, float> &, int,
                       float, float)
      ->smoothing_result<float> =
      &tf::cpp::taubin_smoothed<smoothing_index, float, 3, 3>;
  auto (*taubin_double)(const tf::cpp::mesh<smoothing_index, double> &, int,
                        double, double)
      ->smoothing_result<double> =
      &tf::cpp::taubin_smoothed<smoothing_index, double, 3, 3>;

  CHECK(laplacian_float(float_input.mesh(), 0, 0.5F).points_buffer().size() ==
        4);
  CHECK(laplacian_double(double_input.mesh(), 0, 0.5).points_buffer().size() ==
        4);
  CHECK(taubin_float(float_input.mesh(), 0, 0.5F, -0.53F)
            .points_buffer()
            .size() == 4);
  CHECK(taubin_double(double_input.mesh(), 0, 0.5, -0.53)
            .points_buffer()
            .size() == 4);
}
