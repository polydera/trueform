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
#include "parity_checks.hpp"

#include "trueform/cpp/intersect/async/intersection_curves.hpp"
#include "trueform/cpp/intersect/async/self_intersection_curves.hpp"
#include "trueform/cpp/intersect/intersection_curves.hpp"
#include "trueform/cpp/intersect/self_intersection_curves.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

template <typename Real>
using owned = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

/// A translation along x, as the sixteen numbers a placement is.
template <typename Real> auto x_translation(Real x) -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x,       Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
          Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real>
auto horizontal_triangle(bool transformed = false) -> owned<Real> {
  const auto offset = transformed ? Real{5} : Real{0};
  owned<Real> result{tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {Real{-1} + offset, Real{-1}, Real{0}, Real{1} + offset,
                  Real{-1}, Real{0}, offset, Real{1}, Real{0}})};
  if (transformed)
    result.place(x_translation<Real>(-offset));
  return result;
}

template <typename Real>
auto vertical_triangle(bool transformed = false, Real world_x = Real{0})
    -> owned<Real> {
  const auto offset = transformed ? Real{7} : Real{0};
  owned<Real> result{tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {world_x + offset, Real{-0.5}, Real{-1}, world_x + offset,
                  Real{-0.5}, Real{1}, world_x + offset, Real{0.75}, Real{0}})};
  if (transformed)
    result.place(x_translation<Real>(-offset));
  return result;
}

template <typename Real> auto empty_mesh() -> owned<Real> { return {}; }

template <typename Real> auto self_crossing_mesh() -> owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 4, 5},
      {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0}, Real{0},
       Real{1}, Real{0}, Real{0}, Real{-0.5}, Real{-1}, Real{0}, Real{-0.5},
       Real{1}, Real{0}, Real{0.75}, Real{0}})};
}

struct counting_resolver {
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }
};

} // namespace

TEMPLATE_TEST_CASE("exact intersection covers every transformation pair",
                   "[cpp][intersect][pair][dispatch]", float, double) {
  for (int mask = 0; mask != 4; ++mask) {
    INFO("transformation mask " << mask);
    const auto horizontal = horizontal_triangle<TestType>((mask & 1) != 0);
    const auto vertical = vertical_triangle<TestType>((mask & 2) != 0);

    auto result =
        tf::cpp::intersection_curves(horizontal.mesh(), vertical.mesh());
    CHECK(tf::cpp::test::curves_are_aligned(result, false));
    CHECK(result.size() > 0);
    CHECK(tf::cpp::test::arrays_of(result).points.shape_at(0) > 1);
  }
}

TEMPLATE_TEST_CASE("exact intersection preserves empty miss modes and topology",
                   "[cpp][intersect][pair][empty]", float, double) {
  const auto horizontal = horizontal_triangle<TestType>();
  const auto miss = vertical_triangle<TestType>(false, TestType{4});
  const auto vertical = vertical_triangle<TestType>();
  const auto empty = empty_mesh<TestType>();

  const auto no_hit =
      tf::cpp::intersection_curves(horizontal.mesh(), miss.mesh());
  const auto empty_hit =
      tf::cpp::intersection_curves(horizontal.mesh(), empty.mesh());
  const auto primitive_hit =
      tf::cpp::intersection_curves(horizontal.mesh(), vertical.mesh(),
                                   {tf::intersect_mode::primitives, 0.0});

  CHECK(tf::cpp::test::curves_are_aligned(no_hit, false));
  CHECK(no_hit.size() == 0);
  CHECK(tf::cpp::test::arrays_of(no_hit).points.raw_shape() ==
        tf::small_vector<int, 3>{0, 3});
  CHECK(tf::cpp::test::curves_are_aligned(empty_hit, false));
  CHECK(empty_hit.size() == 0);
  CHECK(tf::cpp::test::curves_are_aligned(primitive_hit, false));
  CHECK(primitive_hit.size() > 0);
}

TEMPLATE_TEST_CASE("multi-domain intersection excludes within-domain work",
                   "[cpp][intersect][list]", float, double) {
  const auto horizontal = horizontal_triangle<TestType>(true);
  const auto vertical = vertical_triangle<TestType>();
  const auto empty = empty_mesh<TestType>();
  const std::vector<typename owned<TestType>::mesh_type> meshes{
      horizontal.mesh(), empty.mesh(), vertical.mesh()};

  const auto result = tf::cpp::intersection_curves(meshes);
  CHECK(tf::cpp::test::curves_are_aligned(result, false));
  CHECK(result.size() > 0);

  const std::vector<typename owned<TestType>::mesh_type> one{horizontal.mesh()};
  const auto one_result = tf::cpp::intersection_curves(one);
  CHECK(tf::cpp::test::curves_are_aligned(one_result, false));
  CHECK(one_result.size() == 0);

  const std::vector<typename owned<TestType>::mesh_type> none;
  CHECK_THROWS_AS(tf::cpp::intersection_curves(none), std::invalid_argument);
}

TEMPLATE_TEST_CASE("self intersection handles hit no-hit and empty meshes",
                   "[cpp][intersect][self]", float, double) {
  const auto crossing = self_crossing_mesh<TestType>();
  const auto triangle = horizontal_triangle<TestType>();
  const auto empty = empty_mesh<TestType>();

  const auto hit = tf::cpp::self_intersection_curves(crossing.mesh());
  const auto no_hit = tf::cpp::self_intersection_curves(triangle.mesh());
  const auto empty_result = tf::cpp::self_intersection_curves(empty.mesh());

  CHECK(tf::cpp::test::curves_are_aligned(hit, false));
  CHECK(hit.size() > 0);
  CHECK(tf::cpp::test::curves_are_aligned(no_hit, false));
  CHECK(no_hit.size() == 0);
  CHECK(tf::cpp::test::curves_are_aligned(empty_result, false));
  CHECK(empty_result.size() == 0);
}

TEMPLATE_TEST_CASE("exact intersection validates forms indices and fields",
                   "[cpp][intersect][validation]", float, double) {
  // a default-constructed owner is the EMPTY mesh, which meets nothing
  const owned<TestType> nothing;
  const auto valid = vertical_triangle<TestType>();
  CHECK(tf::cpp::intersection_curves(nothing.mesh(), valid.mesh()).size() == 0);
  CHECK(tf::cpp::self_intersection_curves(nothing.mesh()).size() == 0);

  // faces that name a point the geometry does not have are refused at the one
  // door the reading passes
  const owned<TestType> malformed{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                      TestType{0}, TestType{0}})};
  CHECK_THROWS_AS(tf::cpp::intersection_curves(malformed.mesh(), valid.mesh()),
                  std::out_of_range);
}

TEMPLATE_TEST_CASE("exact intersection validates configuration before empties",
                   "[cpp][intersect][config]", float, double) {
  const auto owned_a = empty_mesh<TestType>();
  const auto owned_b = empty_mesh<TestType>();
  const auto empty_a = owned_a.mesh();
  const auto empty_b = owned_b.mesh();
  const std::vector<typename owned<TestType>::mesh_type> one_empty{empty_a};

  const auto check_invalid_tolerance = [&](double tolerance) {
    CHECK_THROWS_AS(tf::cpp::intersection_curves(
                        empty_a, empty_b, {tf::intersect_mode::sos, tolerance}),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::intersection_curves(
                        one_empty, {tf::intersect_mode::sos, tolerance}),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::self_intersection_curves(
                        empty_a, {tf::intersect_mode::sos, tolerance}),
                    std::invalid_argument);
  };
  check_invalid_tolerance(-1.0);
  check_invalid_tolerance(std::numeric_limits<double>::infinity());
  check_invalid_tolerance(std::numeric_limits<double>::quiet_NaN());

  if constexpr (std::is_same_v<TestType, float>) {
    CHECK_THROWS_AS(
        tf::cpp::intersection_curves(
            empty_a, empty_b,
            {tf::intersect_mode::sos, std::numeric_limits<double>::max()}),
        std::invalid_argument);
  }

  const auto check_invalid_mode = [&](int mode) {
    const auto config =
        tf::intersect_config{static_cast<tf::intersect_mode>(mode), 0.0};
    CHECK_THROWS_AS(tf::cpp::intersection_curves(empty_a, empty_b, config),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::intersection_curves(one_empty, config),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::self_intersection_curves(empty_a, config),
                    std::invalid_argument);
  };
  check_invalid_mode(0);
  check_invalid_mode(static_cast<int>(tf::intersect_mode::sos) |
                     static_cast<int>(tf::intersect_mode::primitives));
  check_invalid_mode(static_cast<int>(tf::intersect_mode::sos) | 32);
  check_invalid_mode(
      static_cast<int>(tf::intersect_mode::sos | tf::intersect_mode::within));

  for (const auto mode : {
           tf::intersect_mode::sos,
           tf::intersect_mode::primitives,
           tf::intersect_mode::sos | tf::intersect_mode::resolve_contours,
       }) {
    CHECK(tf::cpp::intersection_curves(empty_a, empty_b, {mode, 0.0}).size() ==
          0);
    CHECK(tf::cpp::intersection_curves(one_empty, {mode, 0.0}).size() == 0);
    CHECK(tf::cpp::self_intersection_curves(empty_a, {mode, 0.0}).size() == 0);
  }

  auto invalid_tolerance_future = tf::cpp::async::intersection_curves(
      empty_a, empty_b,
      {tf::intersect_mode::primitives,
       std::numeric_limits<double>::infinity()});
  CHECK_THROWS_AS(invalid_tolerance_future.get(), std::invalid_argument);

  auto invalid_mode_future = tf::cpp::async::self_intersection_curves(
      empty_a, {tf::intersect_mode::sos | tf::intersect_mode::primitives, 0.0});
  CHECK_THROWS_AS(invalid_mode_future.get(), std::invalid_argument);

  auto valid_future = tf::cpp::async::intersection_curves(
      one_empty, {tf::intersect_mode::primitives |
                      tf::intersect_mode::resolve_crossing_contours,
                  0.0});
  CHECK(valid_future.get().size() == 0);
}

TEMPLATE_TEST_CASE("exact intersection async entries read what they borrow",
                   "[cpp][intersect][async]", float, double) {
  const auto horizontal = horizontal_triangle<TestType>();
  const auto vertical = vertical_triangle<TestType>();
  const auto crossing = self_crossing_mesh<TestType>();
  const std::vector<typename owned<TestType>::mesh_type> meshes{
      horizontal.mesh(), vertical.mesh()};

  auto pair_future =
      tf::cpp::async::intersection_curves(horizontal.mesh(), vertical.mesh());
  auto list_future = tf::cpp::async::intersection_curves(meshes);
  auto self_future = tf::cpp::async::self_intersection_curves(crossing.mesh());
  static_assert(
      std::is_same_v<decltype(pair_future),
                     std::future<tf::curves_buffer<tf::cpp::default_index_t,
                                                   TestType, 3>>>);
  static_assert(
      std::is_same_v<decltype(list_future),
                     std::future<tf::curves_buffer<tf::cpp::default_index_t,
                                                   TestType, 3>>>);
  static_assert(
      std::is_same_v<decltype(self_future),
                     std::future<tf::curves_buffer<tf::cpp::default_index_t,
                                                   TestType, 3>>>);

  CHECK(pair_future.get().size() > 0);
  CHECK(list_future.get().size() > 0);
  CHECK(self_future.get().size() > 0);

  auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto custom_a = horizontal_triangle<TestType>();
  const auto custom_b = vertical_triangle<TestType>();
  auto custom_future = tf::cpp::async::intersection_curves(
      counting_resolver{submissions}, custom_a.mesh(), custom_b.mesh());
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom_future.get().size() > 0);

  const owned<TestType> nothing;
  CHECK(tf::cpp::async::self_intersection_curves(nothing.mesh()).get().size() ==
        0);
}

TEMPLATE_TEST_CASE("exact intersection output is deterministic",
                   "[cpp][intersect][determinism]", float, double) {
  const auto horizontal = horizontal_triangle<TestType>(true);
  const auto vertical = vertical_triangle<TestType>(true);
  const auto first =
      tf::cpp::intersection_curves(horizontal.mesh(), vertical.mesh());
  const auto second =
      tf::cpp::intersection_curves(horizontal.mesh(), vertical.mesh());

  CHECK(tf::cpp::test::same_array(tf::cpp::test::arrays_of(first).points,
                                  tf::cpp::test::arrays_of(second).points));
  CHECK(tf::cpp::test::same_array(tf::cpp::test::arrays_of(first).offsets,
                                  tf::cpp::test::arrays_of(second).offsets));
  CHECK(tf::cpp::test::same_array(tf::cpp::test::arrays_of(first).ids,
                                  tf::cpp::test::arrays_of(second).ids));
}
