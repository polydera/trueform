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
#include "trueform/cpp/geometry/async/normals.hpp"
#include "trueform/cpp/geometry/async/point_normals.hpp"
#include "trueform/cpp/geometry/normals.hpp"
#include "trueform/cpp/geometry/point_normals.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <type_traits>

namespace {

template <typename Index, typename Real> struct normals_row {
  using real_type = Real;
  using index_type = Index;
  using owned_type = tf::cpp::test::owned_mesh<Index, Real, 3>;
};

using normals_float_int32 = normals_row<std::int32_t, float>;
using normals_float_int64 = normals_row<std::int64_t, float>;
using normals_double_int32 = normals_row<std::int32_t, double>;
using normals_double_int64 = normals_row<std::int64_t, double>;

template <typename Mesh, typename = void>
struct normals_takes : std::false_type {};

template <typename Mesh>
struct normals_takes<
    Mesh, std::void_t<decltype(tf::cpp::normals(std::declval<const Mesh &>()))>>
    : std::true_type {};

template <typename Mesh, typename = void>
struct point_normals_takes : std::false_type {};

template <typename Mesh>
struct point_normals_takes<Mesh, std::void_t<decltype(tf::cpp::point_normals(
                                     std::declval<const Mesh &>()))>>
    : std::true_type {};

template <typename Mesh, typename = void>
struct async_normals_takes : std::false_type {};

template <typename Mesh>
struct async_normals_takes<Mesh, std::void_t<decltype(tf::cpp::async::normals(
                                     std::declval<const Mesh &>()))>>
    : std::true_type {};

template <typename Mesh, typename = void>
struct async_point_normals_takes : std::false_type {};

template <typename Mesh>
struct async_point_normals_takes<
    Mesh, std::void_t<decltype(tf::cpp::async::point_normals(
              std::declval<const Mesh &>()))>> : std::true_type {};

/// A normal is of a three-dimensional surface by its own definition, so the
/// refusal is the entry's own substitution and never a runtime throw.
static_assert(normals_takes<tf::cpp::mesh<std::int32_t, float, 3>>::value);
static_assert(
    point_normals_takes<tf::cpp::mesh<std::int64_t, double, 3>>::value);
static_assert(
    async_normals_takes<tf::cpp::mesh<std::int64_t, float, 3>>::value);
static_assert(
    async_point_normals_takes<tf::cpp::mesh<std::int32_t, double, 3>>::value);
static_assert(!normals_takes<tf::cpp::mesh<std::int32_t, float, 2>>::value);
static_assert(
    !point_normals_takes<tf::cpp::mesh<std::int64_t, double, 2>>::value);
static_assert(
    !async_normals_takes<tf::cpp::mesh<std::int64_t, float, 2>>::value);
static_assert(
    !async_point_normals_takes<tf::cpp::mesh<std::int32_t, double, 2>>::value);

template <typename Row>
auto python_fixed_fixture() -> typename Row::owned_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  // Exact fixture from python/tests/test_normals.py:create_triangle_mesh_3d.
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 2, 1, 3, 2},
      {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, Real{1.5}, 1, 0})};
}

template <typename Row>
auto python_dynamic_fixture()
    -> tf::cpp::test::mixed_mesh_of<typename Row::owned_type> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  // Exact mixed triangle/quad fixture from python/tests/test_normals.py.
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 3, 7}, {0, 1, 2, 1, 3, 4, 2},
      {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, 2, 0, 0, Real{1.5}, 1, 0})};
}

template <typename Real> auto normals_tolerance() -> double {
  return std::is_same_v<Real, float> ? 2e-6 : 1e-12;
}

template <typename Real>
auto check_unit_z(const tf::cpp::nd_array<Real> &normals, int count) -> void {
  REQUIRE((normals.raw_shape() == tf::small_vector<int, 3>{count, 3}));
  for (int normal = 0; normal < count; ++normal) {
    const auto offset = static_cast<std::size_t>(3 * normal);
    const auto margin = normals_tolerance<Real>();
    CHECK(normals[offset] == Catch::Approx(0.0).margin(margin));
    CHECK(normals[offset + 1] == Catch::Approx(0.0).margin(margin));
    CHECK(normals[offset + 2] == Catch::Approx(1.0).margin(margin));
    const auto length = std::sqrt(
        static_cast<double>(normals[offset] * normals[offset] +
                            normals[offset + 1] * normals[offset + 1] +
                            normals[offset + 2] * normals[offset + 2]));
    CHECK(length == Catch::Approx(1.0).margin(margin));
  }
}

struct normals_counting_resolver {
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }
};

template <typename Row, typename Owned>
auto check_python_normals(Owned &owned, int point_count) -> void {
  using Real = typename Row::real_type;
  // the placement is the caller's, and a normal is read off the raw stored
  // points, so a placed mesh answers exactly as an unplaced one does
  owned.place({Real{2}, Real{0}, Real{0}, Real{7}, Real{0}, Real{3}, Real{0},
               Real{-4}, Real{0}, Real{0}, Real{-5}, Real{9}, Real{0}, Real{0},
               Real{0}, Real{1}});

  check_unit_z(tf::cpp::normals(owned.mesh()), 2);
  // a face normal is the polygon's own, so asking for one states no structure;
  // the membership is what a POINT normal needs, and asking states it
  CHECK_FALSE(owned.cache.is_face_membership_built());
  check_unit_z(tf::cpp::point_normals(owned.mesh()), point_count);
  CHECK(owned.cache.is_face_membership_built());
}

} // namespace

TEMPLATE_TEST_CASE(
    "mesh normal facade matches exact Python fixed and mixed fixtures",
    "[cpp][geometry][normals][matrix][python-parity]", normals_float_int32,
    normals_float_int64, normals_double_int32, normals_double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using function_type =
      tf::cpp::nd_array<Real> (*)(const tf::cpp::mesh<Index, Real, 3, 3> &);
  const function_type face_symbol = &tf::cpp::normals<Index, Real, 3, 3>;
  const function_type point_symbol = &tf::cpp::point_normals<Index, Real, 3, 3>;
  REQUIRE(face_symbol != nullptr);
  REQUIRE(point_symbol != nullptr);

  auto fixed = python_fixed_fixture<TestType>();
  check_python_normals<TestType>(fixed, 4);
  auto mixed = python_dynamic_fixture<TestType>();
  check_python_normals<TestType>(mixed, 5);
}

TEMPLATE_TEST_CASE("mesh face normals match the Python cube direction oracle",
                   "[cpp][geometry][normals][matrix][python-parity][cube]",
                   normals_float_int32, normals_float_int64,
                   normals_double_int32, normals_double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  const typename TestType::owned_type cube{
      tf::cpp::test::polygons_of<Index, Real>(
          {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
           2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5},
          {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
           0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1})};
  const auto values = tf::cpp::normals(cube.mesh());
  const Real expected[12][3] = {{0, 0, -1}, {0, 0, -1}, {0, 0, 1}, {0, 0, 1},
                                {0, -1, 0}, {0, -1, 0}, {0, 1, 0}, {0, 1, 0},
                                {-1, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
  REQUIRE(values.raw_shape() == tf::small_vector<int, 3>{12, 3});
  for (int face = 0; face < 12; ++face)
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      CHECK(values[static_cast<std::size_t>(3 * face + coordinate)] ==
            Catch::Approx(expected[face][coordinate])
                .margin(normals_tolerance<Real>()));
}

TEMPLATE_TEST_CASE("a normal is computed from the reading it was asked for",
                   "[cpp][geometry][normals][matrix][write-through]",
                   normals_float_int32, normals_float_int64,
                   normals_double_int32, normals_double_int64) {
  using Real = typename TestType::real_type;
  auto owned = python_fixed_fixture<TestType>();
  const auto original = tf::cpp::normals(owned.mesh());
  REQUIRE(original[2] == Real{1});

  // the cache remembers a normal for the reading it built it for, so a stated
  // change is what makes the next ask answer the arrays as they now stand
  auto &corners = owned.polygons.faces_buffer().data_buffer();
  const auto first = corners[1];
  corners[1] = corners[2];
  corners[2] = first;
  owned.cache.faces_changed();

  const auto rebuilt = tf::cpp::normals(owned.mesh());
  CHECK(rebuilt.raw_data() != original.raw_data());
  CHECK(rebuilt[2] == Real{-1});
  CHECK(rebuilt[5] == Real{1});
}

TEMPLATE_TEST_CASE("the cache remembers the normals and an entry hands out a "
                   "copy of them",
                   "[cpp][geometry][normals][matrix][cache]",
                   normals_float_int32, normals_float_int64,
                   normals_double_int32, normals_double_int64) {
  using Real = typename TestType::real_type;
  auto owned = python_fixed_fixture<TestType>();
  auto faces = tf::cpp::normals(owned.mesh());
  auto points = tf::cpp::point_normals(owned.mesh());
  CHECK(owned.cache.face_normals_build_count() == 1);
  CHECK(owned.cache.point_normals_build_count() == 1);

  // a second ask costs a stamp check, and a point normal reads the face
  // normals the first ask already built
  static_cast<void>(tf::cpp::normals(owned.mesh()));
  static_cast<void>(tf::cpp::point_normals(owned.mesh()));
  CHECK(owned.cache.face_normals_build_count() == 1);
  CHECK(owned.cache.point_normals_build_count() == 1);

  // what an entry hands out is a COPY, so writing through it reaches nothing
  // another reading of this geometry shares
  faces[2] = Real{-7};
  points[2] = Real{-7};
  CHECK(tf::cpp::normals(owned.mesh())[2] == Real{1});
  CHECK(tf::cpp::point_normals(owned.mesh())[2] == Real{1});
  CHECK(owned.cache.face_normals_build_count() == 1);
  CHECK(owned.cache.point_normals_build_count() == 1);
}

TEMPLATE_TEST_CASE("empty fixed and dynamic meshes return canonical normals",
                   "[cpp][geometry][normals][matrix][empty]",
                   normals_float_int32, normals_float_int64,
                   normals_double_int32, normals_double_int64) {
  const typename TestType::owned_type fixed;
  const tf::cpp::test::mixed_mesh_of<typename TestType::owned_type> mixed;

  for (const auto &value :
       {tf::cpp::normals(fixed.mesh()), tf::cpp::point_normals(fixed.mesh()),
        tf::cpp::normals(mixed.mesh()), tf::cpp::point_normals(mixed.mesh())}) {
    CHECK((value.raw_shape() == tf::small_vector<int, 3>{0, 3}));
    CHECK(value.length() == 0);
  }
}

TEMPLATE_TEST_CASE("async mesh normals answer as the synchronous entry does",
                   "[cpp][geometry][normals][matrix][async]",
                   normals_float_int32, normals_float_int64,
                   normals_double_int32, normals_double_int64) {
  using Real = typename TestType::real_type;
  auto fixed = python_fixed_fixture<TestType>();
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto faces = tf::cpp::async::normals(normals_counting_resolver{submissions},
                                       fixed.mesh());
  static_assert(
      std::is_same_v<decltype(faces), std::future<tf::cpp::nd_array<Real>>>);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  check_unit_z(faces.get(), 2);

  auto points = tf::cpp::async::point_normals(fixed.mesh());
  check_unit_z(points.get(), 4);

  auto mixed = python_dynamic_fixture<TestType>();
  check_unit_z(tf::cpp::async::normals(mixed.mesh()).get(), 2);
  check_unit_z(tf::cpp::async::point_normals(mixed.mesh()).get(), 5);
}

// A default-assembled carrier is the EMPTY mesh: normals answer it with empty
// arrays, here and on the executor.
TEMPLATE_TEST_CASE("the empty mesh's normal operations answer nothing",
                   "[cpp][geometry][normals][empty]", float, double) {
  const tf::cpp::test::owned_mesh<std::int32_t, TestType> empty;
  CHECK(tf::cpp::normals(empty.mesh()).empty());
  CHECK(tf::cpp::point_normals(empty.mesh()).empty());
  CHECK(tf::cpp::async::normals(empty.mesh()).get().empty());
  CHECK(tf::cpp::async::point_normals(empty.mesh()).get().empty());
}
