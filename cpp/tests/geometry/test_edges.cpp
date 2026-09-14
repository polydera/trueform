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

#include "trueform/cpp/geometry/async/sharp_edges.hpp"
#include "trueform/cpp/geometry/sharp_edges.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <type_traits>

namespace {

constexpr long double edges_pi = 3.141592653589793238462643383279502884L;

template <typename Index, typename Real>
auto make_typed_crease_mesh() -> tf::cpp::test::owned_mesh<Index, Real> {
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 2, 1, 0, 3},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{0}, Real{0}, Real{1}})};
}

template <typename Real>
auto make_crease_mesh()
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return make_typed_crease_mesh<tf::cpp::default_index_t, Real>();
}

template <typename Index, typename Real>
auto make_mixed_crease_mesh()
    -> tf::cpp::test::owned_mesh<Index, Real, 3, tf::dynamic_size> {
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 3, 6}, {0, 1, 2, 1, 0, 3},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{0}, Real{0}, Real{1}})};
}

struct edges_counting_resolver {
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

// The edges are named in the mesh's own index width, not in int32: the
// pipeline runs in the types the caller has.
TEMPLATE_TEST_CASE("sharp_edges returns edges in the mesh's own index width",
                   "[cpp][geometry][edges]", float, double) {
  const auto owned = make_crease_mesh<TestType>();

  const auto edges =
      tf::cpp::sharp_edges(owned.mesh(), tf::deg<TestType>{TestType{45}});

  static_assert(
      std::is_same_v<decltype(edges),
                     const tf::cpp::nd_array<tf::cpp::default_index_t>>);
  REQUIRE((edges.raw_shape() == tf::small_vector<int, 3>{1, 2}));
  REQUIRE(edges.length() == 2);
  CHECK(edges[0] == 0);
  CHECK(edges[1] == 1);

  const auto wide = make_typed_crease_mesh<std::int64_t, TestType>();
  const auto wide_edges =
      tf::cpp::sharp_edges(wide.mesh(), tf::deg<TestType>{TestType{45}});
  static_assert(std::is_same_v<decltype(wide_edges),
                               const tf::cpp::nd_array<std::int64_t>>);
  REQUIRE(wide_edges.length() == 2);
  CHECK(wide_edges[0] == 0);
  CHECK(wide_edges[1] == 1);
}

TEMPLATE_TEST_CASE("sharp_edges observes typed angle thresholds",
                   "[cpp][geometry][edges]", float, double) {
  const auto owned = make_crease_mesh<TestType>();

  const auto degrees =
      tf::cpp::sharp_edges(owned.mesh(), tf::deg<TestType>{TestType{60}});
  const auto radians = tf::cpp::sharp_edges(
      owned.mesh(),
      tf::rad<TestType>{static_cast<TestType>(edges_pi) / TestType{3}});
  const auto above = tf::cpp::sharp_edges(
      owned.mesh(),
      tf::rad<TestType>{TestType{3} * static_cast<TestType>(edges_pi) /
                        TestType{4}});

  REQUIRE((degrees.raw_shape() == tf::small_vector<int, 3>{1, 2}));
  REQUIRE(degrees.raw_shape() == radians.raw_shape());
  REQUIRE(degrees.length() == radians.length());
  CHECK(degrees[0] == radians[0]);
  CHECK(degrees[1] == radians[1]);
  CHECK((above.raw_shape() == tf::small_vector<int, 3>{0, 2}));
  CHECK(above.empty());
}

TEMPLATE_TEST_CASE("sharp_edges reuses the manifold-edge cache",
                   "[cpp][geometry][edges]", float, double) {
  const auto owned = make_crease_mesh<TestType>();
  REQUIRE_FALSE(owned.cache.is_manifold_edge_link_built());

  static_cast<void>(
      tf::cpp::sharp_edges(owned.mesh(), tf::deg<TestType>{TestType{45}}));
  REQUIRE(owned.cache.is_manifold_edge_link_fresh(owned.mesh().geometry()));
  REQUIRE(owned.cache.manifold_edge_link_build_count() == 1);

  static_cast<void>(
      tf::cpp::sharp_edges(owned.mesh(), tf::deg<TestType>{TestType{45}}));
  CHECK(owned.cache.is_manifold_edge_link_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.manifold_edge_link_build_count() == 1);
}

TEST_CASE("sharp_edges float and double overloads link from the native archive",
          "[cpp][geometry][edges][link]") {
  const auto float_mesh = make_crease_mesh<float>();
  const auto double_mesh = make_crease_mesh<double>();

  CHECK(
      tf::cpp::sharp_edges(float_mesh.mesh(), tf::deg<float>{45.0F}).length() ==
      2);
  CHECK(tf::cpp::sharp_edges(double_mesh.mesh(), tf::deg<double>{45.0})
            .length() == 2);
}

// A default-assembled carrier is the EMPTY mesh, and an empty mesh has no
// sharp edge.
TEMPLATE_TEST_CASE("sharp_edges answers the empty mesh",
                   "[cpp][geometry][edges][empty]", float, double) {
  const tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType> empty;
  CHECK(tf::cpp::sharp_edges(empty.mesh(), tf::deg<TestType>{TestType{45}})
            .empty());
}

// A dihedral angle is a fact about two faces across an edge, whatever arity
// they are stored at, so a mixed mesh reaches the operation on the same terms
// a triangle one does — synchronously and through an executor.
TEMPLATE_TEST_CASE("sharp_edges takes a mixed mesh at its own arity",
                   "[cpp][geometry][edges][mixed]", float, double) {
  const auto mixed =
      make_mixed_crease_mesh<tf::cpp::default_index_t, TestType>();
  const auto triangles = make_crease_mesh<TestType>();
  const auto expected =
      tf::cpp::sharp_edges(triangles.mesh(), tf::deg<TestType>{TestType{45}});

  const auto edges =
      tf::cpp::sharp_edges(mixed.mesh(), tf::deg<TestType>{TestType{45}});
  REQUIRE(edges.raw_shape() == expected.raw_shape());
  CHECK(edges[0] == expected[0]);
  CHECK(edges[1] == expected[1]);

  auto pending = tf::cpp::async::sharp_edges(mixed.mesh(),
                                             tf::deg<TestType>{TestType{45}});
  CHECK(pending.get().raw_shape() == expected.raw_shape());
}

TEMPLATE_TEST_CASE("async sharp edges preserve their angle overloads",
                   "[cpp][geometry][edges][async]", float, double) {
  const auto owned = make_crease_mesh<TestType>();
  const auto degrees = tf::deg<TestType>{TestType{60}};
  const auto radians = tf::rad<TestType>{static_cast<TestType>(edges_pi) /
                                         static_cast<TestType>(3)};
  const auto expected = tf::cpp::sharp_edges(owned.mesh(), degrees);

  auto degree_result = tf::cpp::async::sharp_edges(owned.mesh(), degrees);
  auto radian_result = tf::cpp::async::sharp_edges(owned.mesh(), radians);
  static_assert(std::is_same_v<decltype(degree_result),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  static_assert(std::is_same_v<decltype(radian_result),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  const auto from_degrees = degree_result.get();
  const auto from_radians = radian_result.get();
  REQUIRE(from_degrees.raw_shape() == expected.raw_shape());
  REQUIRE(from_radians.raw_shape() == expected.raw_shape());
  for (std::size_t index = 0; index < expected.length(); ++index) {
    CHECK(from_degrees[index] == expected[index]);
    CHECK(from_radians[index] == expected[index]);
  }
  CHECK(owned.cache.is_manifold_edge_link_fresh(owned.mesh().geometry()));

  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::sharp_edges(
      edges_counting_resolver{submissions}, owned.mesh(), degrees);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom.get().raw_shape() == expected.raw_shape());

  const tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType> empty;
  CHECK(tf::cpp::async::sharp_edges(empty.mesh(), radians).get().empty());
}
