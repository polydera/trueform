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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/build_manifold_edge_link.hpp"
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
#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Index, typename Real> struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  using owned_type = tf::cpp::test::owned_mesh<Index, Real, 3>;
  using mixed_type =
      tf::cpp::test::owned_mesh<Index, Real, 3, tf::dynamic_size>;
  template <std::size_t Ngon>
  using result_type = tf::polygons_buffer<Index, Real, 3, Ngon>;
};

using float_int32 = matrix_row<std::int32_t, float>;
using float_int64 = matrix_row<std::int64_t, float>;
using double_int32 = matrix_row<std::int32_t, double>;
using double_int64 = matrix_row<std::int64_t, double>;

template <typename Mesh, typename = void>
struct has_positively_oriented : std::false_type {};

template <typename Mesh>
struct has_positively_oriented<
    Mesh, std::void_t<decltype(tf::cpp::positively_oriented(
              std::declval<const Mesh &>()))>> : std::true_type {};

template <typename Mesh, typename = void>
struct has_async_positively_oriented : std::false_type {};

template <typename Mesh>
struct has_async_positively_oriented<
    Mesh, std::void_t<decltype(tf::cpp::async::positively_oriented(
              std::declval<const Mesh &>()))>> : std::true_type {};

static_assert(
    has_positively_oriented<tf::cpp::mesh<std::int32_t, float, 3>>::value);
static_assert(
    has_positively_oriented<tf::cpp::mesh<std::int64_t, double, 3>>::value);
static_assert(
    !has_positively_oriented<tf::cpp::mesh<std::int32_t, float, 2>>::value);
static_assert(
    !has_positively_oriented<tf::cpp::mesh<std::int64_t, double, 2>>::value);
static_assert(has_async_positively_oriented<
              tf::cpp::mesh<std::int64_t, float, 3>>::value);
static_assert(!has_async_positively_oriented<
              tf::cpp::mesh<std::int32_t, double, 2>>::value);
static_assert(!has_positively_oriented<std::string>::value);

template <typename Row>
auto fixed_tetrahedron(int orientation) -> typename Row::owned_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const auto points = {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0},
                       Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1}};
  if (orientation > 0)
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3}, points)};
  if (orientation < 0)
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2}, points)};
  return {tf::cpp::test::polygons_of<Index, Real>(
      {0, 2, 1, 0, 3, 1, 0, 3, 2, 1, 3, 2}, points)};
}

template <typename Row>
auto dynamic_pyramid(int orientation) -> typename Row::mixed_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const auto offsets = {Index{0},  Index{4},  Index{7},
                        Index{10}, Index{13}, Index{16}};
  const auto points = {Real{0}, Real{0}, Real{0},   Real{1},   Real{0},
                       Real{0}, Real{1}, Real{1},   Real{0},   Real{0},
                       Real{1}, Real{0}, Real{0.5}, Real{0.5}, Real{1}};
  if (orientation > 0)
    return {tf::cpp::test::polygons_of<Index, Real>(
        offsets, {0, 3, 2, 1, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4}, points)};
  if (orientation < 0)
    return {tf::cpp::test::polygons_of<Index, Real>(
        offsets, {1, 2, 3, 0, 4, 1, 0, 4, 2, 1, 4, 3, 2, 4, 0, 3}, points)};
  return {tf::cpp::test::polygons_of<Index, Real>(
      offsets, {0, 3, 2, 1, 0, 4, 1, 1, 2, 4, 2, 4, 3, 3, 0, 4}, points)};
}

template <typename Index, typename Real, std::size_t Ngon>
auto measured_volume(const tf::polygons_buffer<Index, Real, 3, Ngon> &value)
    -> Real {
  tf::cpp::cache<Index, Real, 3, Ngon> cache;
  return tf::cpp::signed_volume(tf::cpp::test::reading_over(value, cache));
}

// The layout is the storage's own, so both arities answer through one reader.
template <typename Index, typename Real, std::size_t Ngon>
auto copied_indices(const tf::polygons_buffer<Index, Real, 3, Ngon> &value)
    -> std::vector<Index> {
  const auto &indices = value.faces_buffer().data_buffer();
  return {indices.data(), indices.data() + indices.size()};
}

template <typename Index, typename Real, std::size_t Ngon>
auto has_consistent_edges(
    const tf::polygons_buffer<Index, Real, 3, Ngon> &value) -> bool {
  std::map<std::pair<Index, Index>, int> orientations;
  for (const auto face : value.faces()) {
    auto previous = face.size() - 1;
    for (std::size_t current = 0; current < face.size(); previous = current++) {
      const auto first = face[previous];
      const auto second = face[current];
      orientations[std::minmax(first, second)] += first < second ? 1 : -1;
    }
  }
  return std::all_of(orientations.begin(), orientations.end(),
                     [](const auto &entry) { return entry.second == 0; });
}

template <typename Real> auto translation() -> std::array<Real, 16> {
  return {1, 0, 0, 11, 0, 1, 0, -7, 0, 0, 1, 5, 0, 0, 0, 1};
}

template <typename Row, std::size_t Ngon>
auto check_empty_orientation() -> void {
  for (const bool is_consistent : {false, true}) {
    const tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> input;
    auto output = tf::cpp::positively_oriented(input.mesh(), is_consistent);
    static_assert(std::is_same_v<decltype(output),
                                 typename Row::template result_type<Ngon>>);

    CHECK(output.faces_buffer().size() == 0);
    CHECK(output.points_buffer().size() == 0);
    CHECK(tf::cpp::test::face_indices_of(output).empty());
    if constexpr (Ngon != 3) {
      CHECK(tf::cpp::test::face_offsets_of(output).raw_shape() ==
            tf::small_vector<int, 3>{0});
    }

    // an empty mesh is already oriented, so nothing is asked of it whichever
    // side the caller states
    CHECK_FALSE(input.cache.is_manifold_edge_link_built());
    CHECK_FALSE(input.cache.is_face_membership_built());
  }
}

} // namespace

TEMPLATE_TEST_CASE("positive orientation covers the fixed Python dtype matrix",
                   "[cpp][geometry][orientation][matrix][python-parity][fixed]",
                   float_int32, float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Result = typename TestType::template result_type<3>;
  using function_type =
      Result (*)(const tf::cpp::mesh<Index, Real, 3, 3> &, bool);
  const function_type symbol = &tf::cpp::positively_oriented<Index, Real, 3, 3>;
  REQUIRE(symbol != nullptr);

  auto positive = fixed_tetrahedron<TestType>(1);
  positive.place(translation<Real>());
  const auto positive_before = copied_indices(positive.polygons);
  const auto *points = positive.polygons.points_buffer().data_buffer().data();
  REQUIRE(tf::cpp::signed_volume(positive.mesh()) > Real{0});
  auto positive_result = symbol(positive.mesh(), false);
  static_assert(std::is_same_v<decltype(positive_result), Result>);
  CHECK(copied_indices(positive.polygons) == positive_before);
  CHECK(copied_indices(positive_result) == positive_before);
  CHECK(positive_result.points_buffer().data_buffer().data() != points);
  CHECK(positive.cache.is_manifold_edge_link_fresh(positive.mesh().geometry()));
  CHECK(measured_volume(positive_result) > Real{0});

  auto negative = fixed_tetrahedron<TestType>(-1);
  const auto negative_before = copied_indices(negative.polygons);
  REQUIRE(tf::cpp::signed_volume(negative.mesh()) < Real{0});
  auto negative_result = symbol(negative.mesh(), true);
  CHECK(measured_volume(negative_result) > Real{0});
  CHECK(copied_indices(negative.polygons) == negative_before);
  CHECK(copied_indices(negative_result) != negative_before);
  CHECK_FALSE(negative.cache.is_manifold_edge_link_built());
  CHECK_FALSE(negative.cache.is_face_membership_built());

  auto inconsistent = fixed_tetrahedron<TestType>(0);
  REQUIRE_FALSE(has_consistent_edges(inconsistent.polygons));
  const auto inconsistent_before = copied_indices(inconsistent.polygons);
  tf::cpp::build_manifold_edge_link(inconsistent.mesh());
  REQUIRE(inconsistent.cache.is_manifold_edge_link_fresh(
      inconsistent.mesh().geometry()));
  auto consistent_result = symbol(inconsistent.mesh(), false);
  CHECK(has_consistent_edges(consistent_result));
  CHECK(measured_volume(consistent_result) > Real{0});
  CHECK(copied_indices(inconsistent.polygons) == inconsistent_before);
  CHECK(inconsistent.cache.manifold_edge_link_build_count() == 1);
}

TEMPLATE_TEST_CASE(
    "positive orientation covers dynamic mixed Python carriers",
    "[cpp][geometry][orientation][matrix][python-parity][dynamic]", float_int32,
    float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Result = typename TestType::template result_type<tf::dynamic_size>;

  for (const auto orientation : {1, -1, 0}) {
    const auto input = dynamic_pyramid<TestType>(orientation);
    const auto before = copied_indices(input.polygons);
    const auto *points = input.polygons.points_buffer().data_buffer().data();
    REQUIRE(input.polygons.faces_buffer().size() == 5);
    if (orientation == 0)
      REQUIRE_FALSE(has_consistent_edges(input.polygons));

    auto output =
        tf::cpp::positively_oriented<Index, Real, 3, tf::dynamic_size>(
            input.mesh(), false);
    static_assert(std::is_same_v<decltype(output), Result>);
    CHECK(output.faces_buffer().size() == 5);
    CHECK(tf::cpp::test::face_offsets_of(output).shape_at(0) == 6);
    CHECK(tf::cpp::test::face_offsets_of(output)[1] == Index{4});
    CHECK(has_consistent_edges(output));
    CHECK(measured_volume(output) > Real{0});
    CHECK(copied_indices(input.polygons) == before);
    CHECK(output.points_buffer().data_buffer().data() != points);
  }

  const auto negative = dynamic_pyramid<TestType>(-1);
  const auto before = copied_indices(negative.polygons);
  auto output = tf::cpp::positively_oriented<Index, Real, 3, tf::dynamic_size>(
      negative.mesh(), true);
  CHECK(measured_volume(output) > Real{0});
  CHECK(copied_indices(negative.polygons) == before);
  CHECK(copied_indices(output) != before);
}

TEMPLATE_TEST_CASE(
    "positive orientation returns an independent empty mesh of its own arity",
    "[cpp][geometry][orientation][matrix][empty][fast-path]", float_int32,
    float_int64, double_int32, double_int64) {
  check_empty_orientation<TestType, 3>();
  check_empty_orientation<TestType, tf::dynamic_size>();
}

TEMPLATE_TEST_CASE(
    "positive orientation validates every supported carrier",
    "[cpp][geometry][orientation][matrix][python-parity][validation]",
    float_int32, float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;

  // a default-assembled carrier is the EMPTY mesh and orientation answers it
  const typename TestType::owned_type empty;
  CHECK(tf::cpp::positively_oriented(empty.mesh()).faces_buffer().size() == 0);

  // the door answers for the READING, so a corner that names no point of it is
  // refused at the ask, whatever the storage says on its own
  typename TestType::owned_type beyond;
  beyond.polygons = tf::cpp::test::polygons_of<Index, Real>(
      {0, 1, 4}, {0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK_THROWS_AS(tf::cpp::positively_oriented(beyond.mesh(), true),
                  std::out_of_range);

  auto shrunk = fixed_tetrahedron<TestType>(1);
  auto &coordinates = shrunk.polygons.points_buffer().data_buffer();
  coordinates.allocate(6);
  shrunk.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::positively_oriented(shrunk.mesh(), true),
                  std::out_of_range);

  typename TestType::mixed_type beyond_mixed;
  beyond_mixed.polygons = tf::cpp::test::polygons_of<Index, Real>(
      {0, 3}, {0, 1, 4}, {0, 0, 0, 1, 0, 0, 0, 1, 0});
  CHECK_THROWS_AS(tf::cpp::positively_oriented(beyond_mixed.mesh(), true),
                  std::out_of_range);

  // a block of two indices is a block the storage states happily; it is the
  // MESH's door that says a face needs three
  typename TestType::mixed_type sliver;
  sliver.polygons = tf::cpp::test::polygons_of<Index, Real>({0, 2}, {0, 1},
                                                            {0, 0, 0, 1, 0, 0});
  CHECK_THROWS_AS(tf::cpp::positively_oriented(sliver.mesh(), true),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "async positive orientation takes the full generalized carrier",
    "[cpp][geometry][orientation][matrix][python-parity][async]", float_int32,
    float_int64, double_int32, double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Result = typename TestType::template result_type<tf::dynamic_size>;

  const auto input = dynamic_pyramid<TestType>(-1);
  auto pending =
      tf::cpp::async::positively_oriented<Index, Real, 3, tf::dynamic_size>(
          input.mesh(), true);
  static_assert(std::is_same_v<decltype(pending), std::future<Result>>);
  auto output = pending.get();
  CHECK(measured_volume(output) > Real{0});

  const typename TestType::mixed_type empty;
  CHECK(tf::cpp::async::positively_oriented(empty.mesh(), false)
            .get()
            .faces_buffer()
            .size() == 0);
}

// Orientation is one entry per carrier, so reverse_winding answers for every
// index, real and dimension the matrix carries.
TEST_CASE("orientation symbols are one per carrier",
          "[cpp][geometry][orientation][abi][archive-link]") {
  using narrow = tf::cpp::mesh<std::int32_t, float, 3>;
  using wide = tf::cpp::mesh<std::int64_t, float, 3>;
  using flat = tf::cpp::mesh<std::int32_t, float, 2>;
  using narrow_orientation_function =
      tf::polygons_buffer<std::int32_t, float, 3, 3> (*)(const narrow &, bool);
  using wide_orientation_function =
      tf::polygons_buffer<std::int64_t, float, 3, 3> (*)(const wide &, bool);
  using narrow_reverse_function =
      tf::polygons_buffer<std::int32_t, float, 3, 3> (*)(const narrow &);
  using flat_reverse_function =
      tf::polygons_buffer<std::int32_t, float, 2, 3> (*)(const flat &);
  const narrow_orientation_function narrow_orientation =
      &tf::cpp::positively_oriented<std::int32_t, float, 3, 3>;
  const wide_orientation_function wide_orientation =
      &tf::cpp::positively_oriented<std::int64_t, float, 3, 3>;
  const narrow_reverse_function narrow_reverse =
      &tf::cpp::reverse_winding<std::int32_t, float, 3, 3>;
  const flat_reverse_function flat_reverse =
      &tf::cpp::reverse_winding<std::int32_t, float, 2, 3>;

  CHECK(narrow_orientation != nullptr);
  CHECK(wide_orientation != nullptr);
  CHECK(narrow_reverse != nullptr);
  CHECK(flat_reverse != nullptr);
  CHECK(reinterpret_cast<const void *>(narrow_orientation) !=
        reinterpret_cast<const void *>(wide_orientation));
  CHECK(reinterpret_cast<const void *>(narrow_reverse) !=
        reinterpret_cast<const void *>(flat_reverse));
}
