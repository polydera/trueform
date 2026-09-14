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

#include "trueform/core/buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/cache.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/async/positively_oriented.hpp"
#include "trueform/cpp/geometry/async/reverse_winding.hpp"
#include "trueform/cpp/geometry/positively_oriented.hpp"
#include "trueform/cpp/geometry/reverse_winding.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <future>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using orientation_index = tf::cpp::default_index_t;

template <typename Real>
using orientation_owned = tf::cpp::test::owned_mesh<orientation_index, Real>;

template <typename Real>
using orientation_result = tf::polygons_buffer<orientation_index, Real, 3, 3>;

template <typename Real>
auto make_tetrahedron(bool inconsistent = false) -> orientation_owned<Real> {
  const auto points = {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0},
                       Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1}};
  return {inconsistent ? tf::cpp::test::polygons_of<orientation_index, Real>(
                             {0, 2, 1, 0, 3, 1, 0, 3, 2, 1, 3, 2}, points)
                       : tf::cpp::test::polygons_of<orientation_index, Real>(
                             {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3}, points)};
}

template <typename Real>
auto measured_volume(const orientation_result<Real> &value) -> Real {
  tf::cpp::cache<orientation_index, Real> cache;
  return tf::cpp::signed_volume(tf::cpp::test::reading_over(value, cache));
}

template <typename Real>
auto measured_volume(const orientation_owned<Real> &value) -> Real {
  return tf::cpp::signed_volume(value.mesh());
}

auto has_consistent_edges(const tf::buffer<orientation_index> &faces) -> bool {
  std::map<std::pair<orientation_index, orientation_index>, int> orientations;
  for (std::size_t face = 0; face * 3 < faces.size(); ++face) {
    for (std::size_t edge = 0; edge < 3; ++edge) {
      const auto first = faces[face * 3 + edge];
      const auto second = faces[face * 3 + (edge + 1) % 3];
      orientations[std::minmax(first, second)] += first < second ? 1 : -1;
    }
  }
  for (const auto &entry : orientations)
    if (entry.second != 0)
      return false;
  return true;
}

template <typename Real>
auto corners_of(const orientation_result<Real> &value)
    -> const tf::buffer<orientation_index> & {
  return value.faces_buffer().data_buffer();
}

template <typename Real>
auto corners_of(const orientation_owned<Real> &value)
    -> const tf::buffer<orientation_index> & {
  return value.polygons.faces_buffer().data_buffer();
}

struct orientation_counting_resolver {
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

TEMPLATE_TEST_CASE("reverse winding states arrays of its own",
                   "[cpp][geometry][orientation]", float, double) {
  auto input = make_tetrahedron<TestType>();
  input.place({TestType{1}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
               TestType{1}, TestType{0}, TestType{3}, TestType{0}, TestType{0},
               TestType{1}, TestType{4}, TestType{0}, TestType{0}, TestType{0},
               TestType{1}});
  REQUIRE(measured_volume(input) > TestType{0});

  auto (*operation)(const tf::cpp::mesh<orientation_index, TestType> &)
      ->orientation_result<TestType> =
      &tf::cpp::reverse_winding<orientation_index, TestType, 3, 3>;
  auto output = operation(input.mesh());

  static_assert(std::is_same_v<decltype(output), orientation_result<TestType>>);
  // nothing of the input's storage is shared with the result, and a placement
  // is the caller's to keep rather than the result's to carry
  CHECK(output.points_buffer().data_buffer().data() !=
        input.polygons.points_buffer().data_buffer().data());
  CHECK(corners_of(output).data() != corners_of(input).data());
  CHECK(measured_volume(output) < TestType{0});
  CHECK_FALSE(input.cache.is_manifold_edge_link_built());

  const auto face_count = corners_of(input).size() / 3;
  for (std::size_t face = 0; face < face_count; ++face)
    for (std::size_t index = 0; index < 3; ++index)
      CHECK(corners_of(output)[face * 3 + index] ==
            corners_of(input)[face * 3 + 2 - index]);

  output.points_buffer().data_buffer()[0] = TestType{7};
  CHECK(input.polygons.points_buffer().data_buffer()[0] != TestType{7});
}

TEMPLATE_TEST_CASE("positive orientation fixes consistency and volume",
                   "[cpp][geometry][orientation]", float, double) {
  auto input = make_tetrahedron<TestType>(true);
  input.place({TestType{1}, TestType{0}, TestType{0}, TestType{2}, TestType{0},
               TestType{1}, TestType{0}, TestType{3}, TestType{0}, TestType{0},
               TestType{1}, TestType{4}, TestType{0}, TestType{0}, TestType{0},
               TestType{1}});
  const auto source_corners = std::vector<orientation_index>(
      corners_of(input).begin(), corners_of(input).end());
  REQUIRE_FALSE(has_consistent_edges(corners_of(input)));
  REQUIRE_FALSE(input.cache.is_manifold_edge_link_built());

  auto (*operation)(const tf::cpp::mesh<orientation_index, TestType> &, bool)
      ->orientation_result<TestType> =
      &tf::cpp::positively_oriented<orientation_index, TestType, 3, 3>;
  auto output = operation(input.mesh(), false);

  CHECK(has_consistent_edges(corners_of(output)));
  CHECK(measured_volume(output) > TestType{0});
  // the link is the SOURCE mesh's, because the copy has its connectivity
  CHECK(input.cache.is_manifold_edge_link_fresh(input.mesh().geometry()));
  CHECK(input.cache.is_face_membership_fresh(input.mesh().geometry()));
  CHECK(corners_of(output).data() != corners_of(input).data());
  CHECK(output.points_buffer().data_buffer().data() !=
        input.polygons.points_buffer().data_buffer().data());
  for (std::size_t index = 0; index < source_corners.size(); ++index)
    CHECK(corners_of(input)[index] == source_corners[index]);
}

TEMPLATE_TEST_CASE("positive orientation preserves the consistent fast path",
                   "[cpp][geometry][orientation]", float, double) {
  auto input = make_tetrahedron<TestType>();
  REQUIRE(measured_volume(input) > TestType{0});
  REQUIRE_FALSE(input.cache.is_manifold_edge_link_built());

  auto output = tf::cpp::positively_oriented(input.mesh(), true);

  CHECK_FALSE(input.cache.is_manifold_edge_link_built());
  CHECK(measured_volume(output) > TestType{0});
  for (std::size_t index = 0; index < corners_of(input).size(); ++index)
    CHECK(corners_of(output)[index] == corners_of(input)[index]);
}

// A default-assembled carrier is the EMPTY mesh: orientation answers it with
// an empty result of the same arity.
TEMPLATE_TEST_CASE("orientation answers the empty mesh",
                   "[cpp][geometry][orientation][empty]", float, double) {
  const orientation_owned<TestType> empty;
  CHECK(tf::cpp::positively_oriented(empty.mesh()).faces_buffer().size() == 0);
  CHECK(tf::cpp::reverse_winding(empty.mesh()).faces_buffer().size() == 0);
}

TEMPLATE_TEST_CASE("async orientation answers through the cache it was handed",
                   "[cpp][geometry][orientation][async]", float, double) {
  auto inconsistent = make_tetrahedron<TestType>(true);
  auto positive = tf::cpp::async::positively_oriented(inconsistent.mesh());
  static_assert(std::is_same_v<decltype(positive),
                               std::future<orientation_result<TestType>>>);
  auto positive_result = positive.get();
  CHECK(has_consistent_edges(corners_of(positive_result)));
  CHECK(measured_volume(positive_result) > TestType{0});
  // the job reads the caller's own cache, so what the worker filled is there
  CHECK(inconsistent.cache.is_manifold_edge_link_built());

  auto input = make_tetrahedron<TestType>();
  const auto expected = tf::cpp::reverse_winding(input.mesh());
  auto reversed = tf::cpp::async::reverse_winding(input.mesh());
  static_assert(std::is_same_v<decltype(reversed),
                               std::future<orientation_result<TestType>>>);
  auto reversed_result = reversed.get();
  REQUIRE(corners_of(reversed_result).size() == corners_of(expected).size());
  for (std::size_t index = 0; index < corners_of(expected).size(); ++index)
    CHECK(corners_of(reversed_result)[index] == corners_of(expected)[index]);

  auto consistent = make_tetrahedron<TestType>();
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::positively_oriented(
      orientation_counting_resolver{submissions}, consistent.mesh(), true);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(measured_volume(custom.get()) > TestType{0});
  CHECK_FALSE(consistent.cache.is_manifold_edge_link_built());
  CHECK_FALSE(consistent.cache.is_face_membership_built());

  const orientation_owned<TestType> empty;
  CHECK(tf::cpp::async::reverse_winding(empty.mesh())
            .get()
            .faces_buffer()
            .size() == 0);
}
