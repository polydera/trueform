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
#include "mixed_mesh.hpp"

#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/geometry/async/area.hpp"
#include "trueform/cpp/geometry/async/max_edge_length.hpp"
#include "trueform/cpp/geometry/async/mean_edge_length.hpp"
#include "trueform/cpp/geometry/async/min_edge_length.hpp"
#include "trueform/cpp/geometry/async/signed_volume.hpp"
#include "trueform/cpp/geometry/async/volume.hpp"
#include "trueform/cpp/geometry/make_cylinder_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/max_edge_length.hpp"
#include "trueform/cpp/geometry/mean_edge_length.hpp"
#include "trueform/cpp/geometry/min_edge_length.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/geometry/volume.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

template <typename Index, typename Real, std::size_t Dims> struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  static constexpr auto dims = Dims;
  using owned_type = tf::cpp::test::owned_mesh<Index, Real, Dims>;
  using mixed_type =
      tf::cpp::test::owned_mesh<Index, Real, Dims, tf::dynamic_size>;
};

using float_int32_2d = matrix_row<std::int32_t, float, 2>;
using float_int64_2d = matrix_row<std::int64_t, float, 2>;
using double_int32_2d = matrix_row<std::int32_t, double, 2>;
using double_int64_2d = matrix_row<std::int64_t, double, 2>;
using float_int32_3d = matrix_row<std::int32_t, float, 3>;
using float_int64_3d = matrix_row<std::int64_t, float, 3>;
using double_int32_3d = matrix_row<std::int32_t, double, 3>;
using double_int64_3d = matrix_row<std::int64_t, double, 3>;

template <typename Mesh, typename = void>
struct has_signed_volume : std::false_type {};

template <typename Mesh>
struct has_signed_volume<Mesh, std::void_t<decltype(tf::cpp::signed_volume(
                                   std::declval<const Mesh &>()))>>
    : std::true_type {};

template <typename Mesh, typename = void>
struct has_volume : std::false_type {};

template <typename Mesh>
struct has_volume<
    Mesh, std::void_t<decltype(tf::cpp::volume(std::declval<const Mesh &>()))>>
    : std::true_type {};

template <typename Mesh, typename = void>
struct has_async_signed_volume : std::false_type {};

template <typename Mesh>
struct has_async_signed_volume<
    Mesh, std::void_t<decltype(tf::cpp::async::signed_volume(
              std::declval<const Mesh &>()))>> : std::true_type {};

template <typename Mesh, typename = void>
struct has_async_volume : std::false_type {};

template <typename Mesh>
struct has_async_volume<Mesh, std::void_t<decltype(tf::cpp::async::volume(
                                  std::declval<const Mesh &>()))>>
    : std::true_type {};

static_assert(has_signed_volume<tf::cpp::mesh<std::int32_t, float, 3>>::value);
static_assert(has_signed_volume<tf::cpp::mesh<std::int64_t, double, 3>>::value);
static_assert(!has_signed_volume<tf::cpp::mesh<std::int32_t, float, 2>>::value);
static_assert(
    !has_signed_volume<tf::cpp::mesh<std::int64_t, double, 2>>::value);
static_assert(has_volume<tf::cpp::mesh<std::int32_t, float, 3>>::value);
static_assert(has_volume<tf::cpp::mesh<std::int64_t, double, 3>>::value);
static_assert(!has_volume<tf::cpp::mesh<std::int32_t, float, 2>>::value);
static_assert(!has_volume<tf::cpp::mesh<std::int64_t, double, 2>>::value);
static_assert(
    has_async_signed_volume<tf::cpp::mesh<std::int32_t, float, 3>>::value);
static_assert(
    !has_async_signed_volume<tf::cpp::mesh<std::int32_t, float, 2>>::value);
static_assert(has_async_volume<tf::cpp::mesh<std::int64_t, double, 3>>::value);
static_assert(!has_async_volume<tf::cpp::mesh<std::int64_t, double, 2>>::value);

template <typename Row> auto tolerance() -> double {
  return std::is_same_v<typename Row::real_type, float> ? 1e-5 : 1e-12;
}

template <typename Row> auto fixed_triangle() -> typename Row::owned_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  if constexpr (Row::dims == 2)
    return {tf::cpp::test::polygons_of<Index, Real, 2>({0, 1, 2},
                                                       {0, 0, 1, 0, 0, 1})};
  else
    return {tf::cpp::test::polygons_of<Index, Real, 3>(
        {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
}

template <typename Row>
auto dynamic_triangle_quad() -> typename Row::mixed_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  if constexpr (Row::dims == 2)
    return {tf::cpp::test::polygons_of<Index, Real, 2>(
        {0, 3, 7}, {0, 1, 2, 3, 4, 5, 6},
        {0, 0, 1, 0, 0, 1, 2, 0, 4, 0, 4, 1, 2, 1})};
  else
    return {tf::cpp::test::polygons_of<Index, Real, 3>(
        {0, 3, 7}, {0, 1, 2, 3, 4, 5, 6},
        {0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 4, 0, 0, 4, 1, 0, 2, 1, 0})};
}

template <typename Row>
auto dynamic_box(bool reverse = false) -> typename Row::mixed_type {
  static_assert(Row::dims == 3);
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const auto points = {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0},
                       Real{1}, Real{1}, Real{0}, Real{0}, Real{1}, Real{0},
                       Real{0}, Real{0}, Real{1}, Real{1}, Real{0}, Real{1},
                       Real{1}, Real{1}, Real{1}, Real{0}, Real{1}, Real{1}};
  return {
      reverse
          ? tf::cpp::test::polygons_of<Index, Real, 3>(
                {0, 4, 8, 12, 16, 20, 24}, {1, 2, 3, 0, 7, 6, 5, 4, 4, 5, 1, 0,
                                            5, 6, 2, 1, 6, 7, 3, 2, 7, 4, 0, 3},
                points)
          : tf::cpp::test::polygons_of<Index, Real, 3>(
                {0, 4, 8, 12, 16, 20, 24}, {0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4,
                                            1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7},
                points)};
}

template <typename Row> auto scale_and_translate() {
  using Real = typename Row::real_type;
  if constexpr (Row::dims == 2)
    return std::array<Real, 9>{2, 0, 11, 0, 3, -7, 0, 0, 1};
  else
    return std::array<Real, 16>{2, 0, 0, 11, 0, 3, 0, -7,
                                0, 0, 4, 5,  0, 0, 0, 1};
}

/// The door refuses faces these points do not have, and it answers for the
/// READING, so a mesh whose faces outrun its points is one the caller made by
/// restating its points and saying so.
template <typename Owned> auto shrink_points(Owned &owned) -> void {
  using Real = typename Owned::real_type;
  auto &storage = owned.polygons.points_buffer().data_buffer();
  storage.allocate(2 * Owned::dims);
  for (std::size_t index = 0; index != storage.size(); ++index)
    storage[index] = Real{0};
  storage[Owned::dims] = Real{1};
  owned.cache.points_changed();
}

template <typename Row> auto outrunning_fixed() -> typename Row::owned_type {
  auto value = fixed_triangle<Row>();
  shrink_points(value);
  return value;
}

template <typename Row> auto outrunning_dynamic() -> typename Row::mixed_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  typename Row::mixed_type value;
  if constexpr (Row::dims == 2)
    value.polygons = tf::cpp::test::polygons_of<Index, Real, 2>(
        {0, 3}, {0, 1, 2}, {0, 0, 1, 0, 0, 1});
  else
    value.polygons = tf::cpp::test::polygons_of<Index, Real, 3>(
        {0, 3}, {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0});
  shrink_points(value);
  return value;
}

template <typename Row> auto empty_dynamic() -> typename Row::mixed_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  if constexpr (Row::dims == 2)
    return {tf::cpp::test::polygons_of<Index, Real, 2>({0}, {}, {3, 4})};
  else
    return {tf::cpp::test::polygons_of<Index, Real, 3>({0}, {}, {3, 4, 5})};
}

} // namespace

TEMPLATE_TEST_CASE("mesh measurement fixed and dynamic carrier matrix",
                   "[cpp][geometry][measurements][matrix]", float_int32_2d,
                   float_int64_2d, double_int32_2d, double_int64_2d,
                   float_int32_3d, float_int64_3d, double_int32_3d,
                   double_int64_3d) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Mesh = tf::cpp::mesh<Index, Real, TestType::dims>;
  using function_type = Real (*)(const Mesh &);

  static_assert(
      std::is_same_v<decltype(tf::cpp::area(std::declval<const Mesh &>())),
                     Real>);
  static_assert(std::is_same_v<decltype(tf::cpp::mean_edge_length(
                                   std::declval<const Mesh &>())),
                               Real>);
  const function_type area_symbol = &tf::cpp::area<Index, Real, TestType::dims>;
  const function_type mean_symbol =
      &tf::cpp::mean_edge_length<Index, Real, TestType::dims>;
  REQUIRE(area_symbol != nullptr);
  REQUIRE(mean_symbol != nullptr);

  const auto fixed = fixed_triangle<TestType>();
  CHECK(area_symbol(fixed.mesh()) ==
        Catch::Approx(0.5).margin(tolerance<TestType>()));
  CHECK(mean_symbol(fixed.mesh()) == Catch::Approx((2.0 + std::sqrt(2.0)) / 3.0)
                                         .margin(tolerance<TestType>()));

  const auto dynamic = dynamic_triangle_quad<TestType>();
  CHECK(tf::cpp::area(dynamic.mesh()) ==
        Catch::Approx(2.5).margin(tolerance<TestType>()));
  CHECK(tf::cpp::mean_edge_length(dynamic.mesh()) ==
        Catch::Approx((8.0 + std::sqrt(2.0)) / 7.0)
            .margin(tolerance<TestType>()));
}

TEMPLATE_TEST_CASE(
    "mesh area matches the Python mixed n-gon fixture",
    "[cpp][geometry][measurements][matrix][python-parity][dynamic]",
    float_int32_3d, float_int64_3d, double_int32_3d, double_int64_3d) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  const auto pi = std::acos(-1.0);
  const auto root_three = static_cast<Real>(std::sqrt(3.0));
  const auto point = [](double angle, double axis) -> Real {
    return static_cast<Real>(axis == 0 ? 5.0 + std::cos(angle)
                                       : std::sin(angle));
  };
  const typename TestType::mixed_type value{
      tf::cpp::test::polygons_of<Index, Real, 3, std::initializer_list<Index>,
                                 std::initializer_list<Index>,
                                 std::vector<Real>>(
          {0, 3, 7, 12}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
          std::vector<Real>{0,
                            0,
                            0,
                            1,
                            0,
                            0,
                            Real{0.5},
                            root_three / Real{2},
                            0,
                            2,
                            0,
                            0,
                            3,
                            0,
                            0,
                            3,
                            1,
                            0,
                            2,
                            1,
                            0,
                            point(0.0, 0),
                            point(0.0, 1),
                            0,
                            point(2.0 * pi / 5.0, 0),
                            point(2.0 * pi / 5.0, 1),
                            0,
                            point(4.0 * pi / 5.0, 0),
                            point(4.0 * pi / 5.0, 1),
                            0,
                            point(6.0 * pi / 5.0, 0),
                            point(6.0 * pi / 5.0, 1),
                            0,
                            point(8.0 * pi / 5.0, 0),
                            point(8.0 * pi / 5.0, 1),
                            0})};
  const auto expected =
      std::sqrt(3.0) / 4.0 + 1.0 + (5.0 / 2.0) * std::sin(2.0 * pi / 5.0);
  CHECK(tf::cpp::area(value.mesh()) ==
        Catch::Approx(expected).margin(10.0 * tolerance<TestType>()));
}

TEMPLATE_TEST_CASE(
    "mesh measurements match Python sphere and cylinder reductions",
    "[cpp][geometry][measurements][matrix][python-parity][reduction]",
    float_int32_3d, float_int64_3d, double_int32_3d, double_int64_3d) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  const auto pi = std::acos(-1.0);

  const typename TestType::owned_type sphere{
      tf::cpp::make_sphere_mesh<Index, Real>(Real{2}, 40, 40)};
  CHECK(tf::cpp::area(sphere.mesh()) == Catch::Approx(16.0 * pi).epsilon(0.01));
  CHECK(tf::cpp::signed_volume(sphere.mesh()) > Real{0});
  CHECK(tf::cpp::volume(sphere.mesh()) ==
        Catch::Approx((32.0 / 3.0) * pi).epsilon(0.01));

  const typename TestType::owned_type cylinder{
      tf::cpp::make_cylinder_mesh<Index, Real>(Real{1.5}, Real{3}, 64)};
  CHECK(tf::cpp::area(cylinder.mesh()) ==
        Catch::Approx(13.5 * pi).epsilon(0.02));
  CHECK(tf::cpp::signed_volume(cylinder.mesh()) > Real{0});
  CHECK(tf::cpp::volume(cylinder.mesh()) ==
        Catch::Approx(6.75 * pi).epsilon(0.01));
}

TEMPLATE_TEST_CASE("mesh measurements match dimensional transform fixtures",
                   "[cpp][geometry][measurements][matrix][transform]",
                   float_int32_2d, float_int64_2d, double_int32_2d,
                   double_int64_2d, float_int32_3d, float_int64_3d,
                   double_int32_3d, double_int64_3d) {
  auto owned = [&] {
    if constexpr (TestType::dims == 2)
      return dynamic_triangle_quad<TestType>();
    else
      return dynamic_box<TestType>();
  }();
  owned.place(scale_and_translate<TestType>());
  const auto value = owned.mesh();

  if constexpr (TestType::dims == 2) {
    CHECK(tf::cpp::area(value) ==
          Catch::Approx(15.0).margin(tolerance<TestType>()));
    CHECK(tf::cpp::mean_edge_length(value) ==
          Catch::Approx((19.0 + std::sqrt(13.0)) / 7.0)
              .margin(tolerance<TestType>()));
  } else {
    CHECK(tf::cpp::area(value) ==
          Catch::Approx(52.0).margin(tolerance<TestType>()));
    CHECK(tf::cpp::signed_volume(value) ==
          Catch::Approx(24.0).margin(tolerance<TestType>()));
    CHECK(tf::cpp::volume(value) ==
          Catch::Approx(24.0).margin(tolerance<TestType>()));
    CHECK(tf::cpp::mean_edge_length(value) ==
          Catch::Approx(3.0).margin(tolerance<TestType>()));
  }
}

TEMPLATE_TEST_CASE("3D mesh volume preserves winding and dynamic polygons",
                   "[cpp][geometry][measurements][matrix][volume][winding]",
                   float_int32_3d, float_int64_3d, double_int32_3d,
                   double_int64_3d) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Mesh = tf::cpp::mesh<Index, Real, TestType::dims, tf::dynamic_size>;
  using function_type = Real (*)(const Mesh &);
  const function_type signed_symbol =
      &tf::cpp::signed_volume<Index, Real, TestType::dims, tf::dynamic_size>;
  const function_type volume_symbol =
      &tf::cpp::volume<Index, Real, TestType::dims, tf::dynamic_size>;
  REQUIRE(signed_symbol != nullptr);
  REQUIRE(volume_symbol != nullptr);

  const auto outward = dynamic_box<TestType>();
  const auto inward = dynamic_box<TestType>(true);
  CHECK(signed_symbol(outward.mesh()) ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));
  CHECK(signed_symbol(inward.mesh()) ==
        Catch::Approx(-1.0).margin(tolerance<TestType>()));
  CHECK(volume_symbol(outward.mesh()) ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));
  CHECK(volume_symbol(inward.mesh()) ==
        Catch::Approx(1.0).margin(tolerance<TestType>()));
}

// A mesh with no faces measures zero, whatever built it and whatever arity it
// states: a default-assembled carrier is the empty mesh and not a state that
// is refused.
TEMPLATE_TEST_CASE("mesh measurements define the empty answer",
                   "[cpp][geometry][measurements][matrix][empty]",
                   float_int32_2d, float_int64_2d, double_int32_2d,
                   double_int64_2d, float_int32_3d, float_int64_3d,
                   double_int32_3d, double_int64_3d) {
  using Real = typename TestType::real_type;
  const auto measures_nothing = [](const auto &value) {
    CHECK(tf::cpp::area(value) == Real{0});
    CHECK(tf::cpp::mean_edge_length(value) == Real{0});
    CHECK(tf::cpp::min_edge_length(value) == Real{0});
    CHECK(tf::cpp::max_edge_length(value) == Real{0});
    if constexpr (TestType::dims == 3) {
      CHECK(tf::cpp::signed_volume(value) == Real{0});
      CHECK(tf::cpp::volume(value) == Real{0});
    }
  };
  measures_nothing(typename TestType::owned_type{}.mesh());
  measures_nothing(empty_dynamic<TestType>().mesh());
}

TEMPLATE_TEST_CASE("async mesh measurements take every matrix carrier",
                   "[cpp][geometry][measurements][matrix][async]",
                   float_int32_2d, float_int64_2d, double_int32_2d,
                   double_int64_2d, float_int32_3d, float_int64_3d,
                   double_int32_3d, double_int64_3d) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  const auto owned = [&] {
    if constexpr (TestType::dims == 2)
      return dynamic_triangle_quad<TestType>();
    else
      return dynamic_box<TestType>();
  }();

  auto area =
      tf::cpp::async::area<Index, Real, TestType::dims, tf::dynamic_size>(
          owned.mesh());
  auto mean = tf::cpp::async::mean_edge_length<Index, Real, TestType::dims,
                                               tf::dynamic_size>(owned.mesh());
  static_assert(std::is_same_v<decltype(area), std::future<Real>>);
  static_assert(std::is_same_v<decltype(mean), std::future<Real>>);

  if constexpr (TestType::dims == 3) {
    auto signed_result =
        tf::cpp::async::signed_volume<Index, Real, TestType::dims,
                                      tf::dynamic_size>(owned.mesh());
    auto absolute_result =
        tf::cpp::async::volume<Index, Real, TestType::dims, tf::dynamic_size>(
            owned.mesh());
    CHECK(area.get() == Catch::Approx(6.0).margin(tolerance<TestType>()));
    CHECK(mean.get() == Catch::Approx(1.0).margin(tolerance<TestType>()));
    CHECK(signed_result.get() ==
          Catch::Approx(1.0).margin(tolerance<TestType>()));
    CHECK(absolute_result.get() ==
          Catch::Approx(1.0).margin(tolerance<TestType>()));
  } else {
    CHECK(area.get() == Catch::Approx(2.5).margin(tolerance<TestType>()));
    CHECK(mean.get() == Catch::Approx((8.0 + std::sqrt(2.0)) / 7.0)
                            .margin(tolerance<TestType>()));
  }
}

TEMPLATE_TEST_CASE(
    "mesh measurements reject unsafe fixed and dynamic face indices",
    "[cpp][geometry][measurements][matrix][validation][indices]",
    float_int32_2d, float_int64_2d, double_int32_2d, double_int64_2d,
    float_int32_3d, float_int64_3d, double_int32_3d, double_int64_3d) {
  using Index = typename TestType::index_type;

  // the arity is the carrier's own type, so each is refused on its own
  const auto refuses = [](const auto &value) {
    CHECK_THROWS_AS(tf::cpp::area(value), std::out_of_range);
    CHECK_THROWS_AS(tf::cpp::mean_edge_length(value), std::out_of_range);
    if constexpr (TestType::dims == 3) {
      CHECK_THROWS_AS(tf::cpp::signed_volume(value), std::out_of_range);
      CHECK_THROWS_AS(tf::cpp::volume(value), std::out_of_range);
    }
  };
  static_cast<void>(sizeof(Index));
  refuses(outrunning_fixed<TestType>().mesh());
  const auto dynamic_greater = outrunning_dynamic<TestType>();
  refuses(dynamic_greater.mesh());

  auto pending =
      tf::cpp::async::area<Index, typename TestType::real_type, TestType::dims,
                           tf::dynamic_size>(dynamic_greater.mesh());
  CHECK_THROWS_AS(pending.get(), std::out_of_range);
}

// Faces are refused at the READING, which is the one door: a corner that names
// no point of it is refused however the storage came to say so.
TEST_CASE("edge extrema refuse faces their points do not have",
          "[cpp][geometry][measurements][validation][indices]") {
  tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float> beyond;
  beyond.polygons = tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
      {0, 1, 3}, {0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK_THROWS_AS(tf::cpp::min_edge_length(beyond.mesh()), std::out_of_range);

  tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float> negative;
  negative.polygons =
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
          {0, 1, -1}, {0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK_THROWS_AS(tf::cpp::max_edge_length(negative.mesh()), std::out_of_range);

  auto shrunk = fixed_triangle<float_int32_3d>();
  shrink_points(shrunk);
  CHECK_THROWS_AS(tf::cpp::min_edge_length(shrunk.mesh()), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::max_edge_length(shrunk.mesh()), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::async::min_edge_length(shrunk.mesh()).get(),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::async::max_edge_length(shrunk.mesh()).get(),
                  std::out_of_range);
}

// A mesh is one READING of the geometry, so a measurement answers for the
// arrays that reading names and for no others.
TEST_CASE("a measurement reads the mesh it was handed",
          "[cpp][geometry][measurements][validation][reading]") {
  auto original = fixed_triangle<float_int32_3d>();
  const auto before = original.mesh();
  CHECK(tf::cpp::area(before) == Catch::Approx(0.5));

  auto doubled = fixed_triangle<float_int32_3d>();
  auto &coordinates = doubled.polygons.points_buffer().data_buffer();
  for (std::size_t index = 0; index != coordinates.size(); ++index)
    coordinates[index] *= 2.0F;
  doubled.cache.points_changed();

  CHECK(tf::cpp::area(doubled.mesh()) == Catch::Approx(2.0));
  CHECK(tf::cpp::area(original.mesh()) == Catch::Approx(0.5));
  CHECK(tf::cpp::async::area(original.mesh()).get() == Catch::Approx(0.5));
}

// One operation is one entry, its compiled symbol takes the mesh, and each
// arity is its own instantiation.
TEST_CASE("mesh measurements are addressable at each arity",
          "[cpp][geometry][measurements][abi][archive-link]") {
  using triangle_mesh = tf::cpp::mesh<std::int32_t, float, 3, 3>;
  using dynamic_mesh = tf::cpp::mesh<std::int32_t, float, 3, tf::dynamic_size>;
  using triangle_function = float (*)(const triangle_mesh &);
  using dynamic_function = float (*)(const dynamic_mesh &);

  const triangle_function triangle_area =
      &tf::cpp::area<std::int32_t, float, 3, 3>;
  const dynamic_function dynamic_area =
      &tf::cpp::area<std::int32_t, float, 3, tf::dynamic_size>;
  const triangle_function triangle_signed_volume =
      &tf::cpp::signed_volume<std::int32_t, float, 3, 3>;
  const triangle_function triangle_mean =
      &tf::cpp::mean_edge_length<std::int32_t, float, 3, 3>;
  const triangle_function triangle_minimum =
      &tf::cpp::min_edge_length<std::int32_t, float, 3, 3>;
  const triangle_function triangle_maximum =
      &tf::cpp::max_edge_length<std::int32_t, float, 3, 3>;
  const triangle_function triangle_volume =
      &tf::cpp::volume<std::int32_t, float, 3, 3>;

  CHECK(triangle_area != nullptr);
  CHECK(dynamic_area != nullptr);
  CHECK(triangle_signed_volume != nullptr);
  CHECK(triangle_mean != nullptr);
  CHECK(triangle_minimum != nullptr);
  CHECK(triangle_maximum != nullptr);
  CHECK(triangle_volume != nullptr);
  CHECK(reinterpret_cast<const void *>(triangle_area) !=
        reinterpret_cast<const void *>(dynamic_area));
}
