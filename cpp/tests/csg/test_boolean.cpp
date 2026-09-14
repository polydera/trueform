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
#include "nd_array.hpp"
#include "trueform/cpp/csg/async/make_boolean.hpp"
#include "trueform/cpp/csg/make_boolean.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

/// The sixteen numbers a placement is, which is what an owner is placed by.
template <typename Real>
auto translation(Real x, Real y = Real{0}, Real z = Real{0})
    -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x, Real{0}, Real{1}, Real{0}, y,
          Real{0}, Real{0}, Real{1}, z, Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Index, typename Real, std::size_t Ngon = 3>
using boolean_owned = tf::cpp::test::owned_mesh<Index, Real, 3, Ngon>;

template <typename Real>
auto overlapping_boxes()
    -> std::pair<boolean_owned<tf::cpp::default_index_t, Real>,
                 boolean_owned<tf::cpp::default_index_t, Real>> {
  boolean_owned<tf::cpp::default_index_t, Real> a{
      tf::cpp::make_box_mesh(Real{2}, Real{2}, Real{2})};
  boolean_owned<tf::cpp::default_index_t, Real> b{
      tf::cpp::make_box_mesh(Real{2}, Real{2}, Real{2})};
  b.place(translation<Real>(Real{1}));
  return {std::move(a), std::move(b)};
}

template <typename Real>
auto empty_mesh() -> boolean_owned<tf::cpp::default_index_t, Real> {
  return {};
}

/// The layout is the geometry's own, so a fixture states the arity it wants
/// and the triangles are restated as the blocks that say the same faces.
template <std::size_t Ngon, typename Index, typename Real>
auto at_arity(tf::polygons_buffer<Index, Real, 3, 3> value)
    -> boolean_owned<Index, Real, Ngon> {
  if constexpr (Ngon == 3) {
    return {std::move(value)};
  } else {
    tf::buffer<Index> offsets;
    offsets.allocate(value.size() + 1);
    for (std::size_t face = 0; face != offsets.size(); ++face)
      offsets[face] = static_cast<Index>(face * 3);
    return {tf::cpp::test::polygons_of<Index, Real>(
        offsets, value.faces_buffer().data_buffer(),
        value.points_buffer().data_buffer())};
  }
}

template <typename Index, typename Real, std::size_t Ngon = 3>
auto typed_box(Real width = Real{2}) -> boolean_owned<Index, Real, Ngon> {
  return at_arity<Ngon>(
      tf::cpp::make_box_mesh<Index, Real>(width, width, width));
}

template <typename Index, typename Real, std::size_t Ngon = 3>
auto typed_empty_mesh() -> boolean_owned<Index, Real, Ngon> {
  return {};
}

template <typename Index, typename Real, std::size_t Ngon>
auto coherent(const tf::polygons_buffer<Index, Real, 3, Ngon> &value,
              const tf::cpp::nd_array<Index> &face_labels) -> bool {
  return face_labels.is_valid() && face_labels.ndim() == 1 &&
         face_labels.size() == static_cast<int>(value.size());
}

template <typename Index, typename Real>
auto coherent(const tf::curves_buffer<Index, Real, 3> &value) -> bool {
  const auto arrays = tf::cpp::test::arrays_of(value);
  return arrays.points.raw_shape().size() == 2 &&
         arrays.points.shape_at(1) == 3 &&
         arrays.offsets.length() ==
             static_cast<std::size_t>(value.size()) + (value.size() ? 1 : 0);
}

template <typename Result>
auto boolean_metadata_in_range(const Result &result, int faces_a, int faces_b)
    -> bool {
  const auto faces = static_cast<int>(result.mesh.size());
  if (result.labels.size() != faces || result.face_labels.size() != faces)
    return false;
  for (int index = 0; index < faces; ++index) {
    const auto source = result.labels[static_cast<std::size_t>(index)];
    const auto face = result.face_labels[static_cast<std::size_t>(index)];
    if (source != 0 && source != 1)
      return false;
    const auto source_size = source == 0 ? faces_a : faces_b;
    if (face < 0 || face >= source_size)
      return false;
  }
  return true;
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

struct mutating_resolver {
  std::shared_ptr<std::atomic<int>> submissions;
  std::function<void()> mutation;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    mutation();
    return std::make_shared<state_type<T>>();
  }
};

template <typename Index0, typename Real, typename Index1,
          std::size_t Ngon0 = 3, std::size_t Ngon1 = 3>
auto check_typed_pair() -> void {
  using OutputIndex = tf::cpp::common_index_t<Index0, Index1>;
  constexpr auto OutputNgon =
      tf::cpp::detail::concatenated_arity_v<Ngon0, Ngon1>;
  const auto owned_a = typed_box<Index0, Real, Ngon0>();
  auto owned_b = typed_box<Index1, Real, Ngon1>();
  owned_b.place(translation<Real>(Real{1}));
  const auto a = owned_a.mesh();
  const auto b = owned_b.mesh();

  const auto boolean =
      tf::cpp::make_boolean<Index0, Real, Index1>(a, b, tf::boolean_op::merge);
  static_assert(
      std::is_same_v<std::decay_t<decltype(boolean)>,
                     tf::cpp::boolean_result<OutputIndex, Real, OutputNgon>>);
  // an uncut face is emitted verbatim, so the result states the arity that
  // can state both operands
  STATIC_REQUIRE(
      std::is_same_v<std::decay_t<decltype(boolean.mesh)>,
                     tf::polygons_buffer<OutputIndex, Real, 3, OutputNgon>>);
  CHECK(boolean.mesh.size() > 0);
  CHECK(boolean_metadata_in_range(boolean,
                                  static_cast<int>(owned_a.polygons.size()),
                                  static_cast<int>(owned_b.polygons.size())));

  const auto boolean_curves =
      tf::cpp::make_boolean_with_curves<Index0, Real, Index1>(
          a, b, tf::boolean_op::intersection);
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(boolean_curves)>,
          tf::cpp::boolean_with_curves_result<OutputIndex, Real, OutputNgon>>);
  CHECK(boolean_metadata_in_range(boolean_curves,
                                  static_cast<int>(owned_a.polygons.size()),
                                  static_cast<int>(owned_b.polygons.size())));
  CHECK(coherent(boolean_curves.curves));
  CHECK(boolean_curves.curves.size() > 0);
}

} // namespace

TEMPLATE_TEST_CASE("boolean facade covers operations configs and results",
                   "[cpp][csg][boolean]", float, double) {
  for (const auto operation :
       {tf::boolean_op::merge, tf::boolean_op::intersection,
        tf::boolean_op::left_difference, tf::boolean_op::right_difference}) {
    const auto [a, b] = overlapping_boxes<TestType>();
    const auto result = tf::cpp::make_boolean(a.mesh(), b.mesh(), operation);
    CHECK(result.mesh.size() > 0);
    CHECK(result.labels.is_valid());
    CHECK(result.labels.size() == static_cast<int>(result.mesh.size()));
    CHECK(coherent(result.mesh, result.face_labels));
  }

  const auto [sheet_a, sheet_b] = overlapping_boxes<TestType>();
  const auto against_sheet = tf::cpp::make_boolean(
      sheet_a.mesh(), sheet_b.mesh(), tf::boolean_op::left_difference, {1});
  CHECK(against_sheet.mesh.size() > 0);
  CHECK(coherent(against_sheet.mesh, against_sheet.face_labels));

  const auto [a, b] = overlapping_boxes<TestType>();
  const auto result = tf::cpp::make_boolean_with_curves(a.mesh(), b.mesh(),
                                                        tf::boolean_op::merge);
  CHECK(result.mesh.size() > 0);
  CHECK(result.labels.size() == static_cast<int>(result.mesh.size()));
  CHECK(coherent(result.mesh, result.face_labels));
  CHECK(coherent(result.curves));
  CHECK(result.curves.size() > 0);
}

TEMPLATE_TEST_CASE("cut facade applies pair transformations",
                   "[cpp][csg][transform]", float, double) {
  const auto [a, b] = overlapping_boxes<TestType>();
  const auto intersection =
      tf::cpp::make_boolean(a.mesh(), b.mesh(), tf::boolean_op::intersection);
  auto min_x = std::numeric_limits<TestType>::infinity();
  auto max_x = -std::numeric_limits<TestType>::infinity();
  for (const auto point : intersection.mesh.points()) {
    min_x = std::min(min_x, point[0]);
    max_x = std::max(max_x, point[0]);
  }
  CHECK(min_x >= TestType{-1e-4});
  CHECK(max_x <= TestType{1.0001});
}

TEMPLATE_TEST_CASE("cut facade validates operations configs and indices",
                   "[cpp][csg][validation]", float, double) {
  const auto [owned_a, owned_b] = overlapping_boxes<TestType>();
  const auto a = owned_a.mesh();
  const auto b = owned_b.mesh();
  CHECK_THROWS_AS(tf::cpp::make_boolean(a, b, static_cast<tf::boolean_op>(99)),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_boolean(a, b, tf::boolean_op::merge, {2}),
                  std::out_of_range);

  auto bad_config = tf::arrangement_config{};
  bad_config.intersect.tolerance = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(
      tf::cpp::make_boolean(a, b, tf::boolean_op::merge, {}, bad_config),
      std::invalid_argument);
  bad_config = {};
  bad_config.intersect.mode = static_cast<tf::intersect_mode>(1024);
  CHECK_THROWS_AS(
      tf::cpp::make_boolean(a, b, tf::boolean_op::merge, {}, bad_config),
      std::invalid_argument);
  bad_config = {};
  bad_config.triangulation = static_cast<tf::triangulation_type>(2);
  CHECK_THROWS_AS(
      tf::cpp::make_boolean(a, b, tf::boolean_op::merge, {}, bad_config),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_boolean_with_curves(a, b, tf::boolean_op::merge,
                                                    {}, bad_config),
                  std::invalid_argument);

  // a face naming a point the geometry does not have is refused at the one
  // door the reading passes
  const boolean_owned<tf::cpp::default_index_t, TestType> malformed{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                      TestType{0}, TestType{0}})};
  CHECK_THROWS_AS(
      tf::cpp::make_boolean(malformed.mesh(), b, tf::boolean_op::merge),
      std::out_of_range);
}

TEMPLATE_TEST_CASE("cut facade preserves coherent empty results",
                   "[cpp][csg][empty]", float, double) {
  const auto empty_a = empty_mesh<TestType>();
  const auto empty_b = empty_mesh<TestType>();
  const auto boolean = tf::cpp::make_boolean(empty_a.mesh(), empty_b.mesh(),
                                             tf::boolean_op::merge);
  CHECK(boolean.mesh.size() == 0);
  CHECK(boolean.labels.size() == 0);
  CHECK(coherent(boolean.mesh, boolean.face_labels));
}

TEMPLATE_TEST_CASE("cut async entries read the operands they borrow",
                   "[cpp][csg][async]", float, double) {
  const auto [boolean_a, boolean_b] = overlapping_boxes<TestType>();
  const auto [curves_a, curves_b] = overlapping_boxes<TestType>();

  auto boolean_future = tf::cpp::async::make_boolean(
      boolean_a.mesh(), boolean_b.mesh(), tf::boolean_op::merge);
  auto boolean_curves_future = tf::cpp::async::make_boolean_with_curves(
      curves_a.mesh(), curves_b.mesh(), tf::boolean_op::intersection);

  static_assert(std::is_same_v<decltype(boolean_future),
                               std::future<tf::cpp::boolean_result<
                                   tf::cpp::default_index_t, TestType, 3>>>);
  static_assert(std::is_same_v<decltype(boolean_curves_future),
                               std::future<tf::cpp::boolean_with_curves_result<
                                   tf::cpp::default_index_t, TestType, 3>>>);

  const auto boolean = boolean_future.get();
  const auto boolean_curves = boolean_curves_future.get();
  CHECK(coherent(boolean.mesh, boolean.face_labels));
  CHECK(coherent(boolean_curves.curves));

  auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto [custom_a, custom_b] = overlapping_boxes<TestType>();
  auto custom_future = tf::cpp::async::make_boolean(
      counting_resolver{submissions}, custom_a.mesh(), custom_b.mesh(),
      tf::boolean_op::left_difference);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom_future.get().mesh.size() > 0);
}

TEMPLATE_TEST_CASE("cut review covers every boolean result and metadata value",
                   "[cpp][csg][review]", float, double) {
  for (const auto operation :
       {tf::boolean_op::merge, tf::boolean_op::intersection,
        tf::boolean_op::left_difference, tf::boolean_op::right_difference}) {
    const auto [plain_a, plain_b] = overlapping_boxes<TestType>();
    const auto plain =
        tf::cpp::make_boolean(plain_a.mesh(), plain_b.mesh(), operation);
    CHECK(plain.mesh.size() > 0);
    CHECK(boolean_metadata_in_range(plain,
                                    static_cast<int>(plain_a.polygons.size()),
                                    static_cast<int>(plain_b.polygons.size())));

    const auto [curves_a, curves_b] = overlapping_boxes<TestType>();
    const auto with_curves = tf::cpp::make_boolean_with_curves(
        curves_a.mesh(), curves_b.mesh(), operation);
    CHECK(with_curves.mesh.size() > 0);
    CHECK(boolean_metadata_in_range(
        with_curves, static_cast<int>(curves_a.polygons.size()),
        static_cast<int>(curves_b.polygons.size())));
    CHECK(coherent(with_curves.curves));
    CHECK(with_curves.curves.size() > 0);
  }
}

TEMPLATE_TEST_CASE("cut review covers transformed first and self behavior",
                   "[cpp][csg][review]", float, double) {
  boolean_owned<tf::cpp::default_index_t, TestType> first{
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2})};
  first.place(translation<TestType>(TestType{1}));
  const boolean_owned<tf::cpp::default_index_t, TestType> second{
      tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2})};

  const auto intersection = tf::cpp::make_boolean(first.mesh(), second.mesh(),
                                                  tf::boolean_op::intersection);
  REQUIRE(intersection.mesh.size() > 0);
  auto boolean_min_x = std::numeric_limits<TestType>::infinity();
  auto boolean_max_x = -std::numeric_limits<TestType>::infinity();
  for (const auto point : intersection.mesh.points()) {
    boolean_min_x = std::min(boolean_min_x, point[0]);
    boolean_max_x = std::max(boolean_max_x, point[0]);
  }
  // the placement moved the operand's coordinates into the result
  CHECK(std::abs(boolean_min_x - TestType{0}) <= TestType{1e-4});
  CHECK(std::abs(boolean_max_x - TestType{1}) <= TestType{1e-4});

  const auto intersection_with_curves = tf::cpp::make_boolean_with_curves(
      first.mesh(), second.mesh(), tf::boolean_op::intersection);
  REQUIRE(intersection_with_curves.mesh.size() > 0);
  auto curves_min_x = std::numeric_limits<TestType>::infinity();
  auto curves_max_x = -std::numeric_limits<TestType>::infinity();
  for (const auto point : intersection_with_curves.mesh.points()) {
    curves_min_x = std::min(curves_min_x, point[0]);
    curves_max_x = std::max(curves_max_x, point[0]);
  }
  CHECK(std::abs(curves_min_x - TestType{0}) <= TestType{1e-4});
  CHECK(std::abs(curves_max_x - TestType{1}) <= TestType{1e-4});
  CHECK(intersection_with_curves.curves.size() > 0);
}

TEMPLATE_TEST_CASE("cut review covers empty ordering and result variants",
                   "[cpp][csg][review]", float, double) {
  const auto check_boolean_order = [](bool first_nonempty,
                                      tf::boolean_op operation,
                                      int expected_faces, int expected_source) {
    const auto box = boolean_owned<tf::cpp::default_index_t, TestType>{
        tf::cpp::make_box_mesh(TestType{2}, TestType{2}, TestType{2})};
    const auto nothing = empty_mesh<TestType>();
    const auto &first = first_nonempty ? box : nothing;
    const auto &second = first_nonempty ? nothing : box;
    const auto plain =
        tf::cpp::make_boolean(first.mesh(), second.mesh(), operation);
    CHECK(plain.mesh.size() == static_cast<std::size_t>(expected_faces));
    CHECK(boolean_metadata_in_range(plain,
                                    static_cast<int>(first.polygons.size()),
                                    static_cast<int>(second.polygons.size())));
    for (const auto source : plain.labels.make_range())
      CHECK(source == expected_source);

    const auto with_curves = tf::cpp::make_boolean_with_curves(
        first.mesh(), second.mesh(), operation);
    CHECK(with_curves.mesh.size() == static_cast<std::size_t>(expected_faces));
    CHECK(boolean_metadata_in_range(with_curves,
                                    static_cast<int>(first.polygons.size()),
                                    static_cast<int>(second.polygons.size())));
    CHECK(with_curves.curves.size() == 0);
    for (const auto source : with_curves.labels.make_range())
      CHECK(source == expected_source);
  };

  constexpr auto box_faces = 12;
  check_boolean_order(true, tf::boolean_op::merge, box_faces, 0);
  check_boolean_order(false, tf::boolean_op::merge, box_faces, 1);
  check_boolean_order(true, tf::boolean_op::intersection, 0, 0);
  check_boolean_order(false, tf::boolean_op::intersection, 0, 0);
  check_boolean_order(true, tf::boolean_op::left_difference, box_faces, 0);
  check_boolean_order(false, tf::boolean_op::left_difference, 0, 0);
  check_boolean_order(true, tf::boolean_op::right_difference, 0, 0);
  check_boolean_order(false, tf::boolean_op::right_difference, box_faces, 1);
}

TEMPLATE_TEST_CASE("cut review covers async borrowing and one submission",
                   "[cpp][csg][review]", float, double) {
  auto submissions = std::make_shared<std::atomic<int>>(0);

  const auto [boolean_a, boolean_b] = overlapping_boxes<TestType>();
  auto boolean = tf::cpp::async::make_boolean(
      counting_resolver{submissions}, boolean_a.mesh(), boolean_b.mesh(),
      tf::boolean_op::merge);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(boolean.get().mesh.size() > 0);

  const auto [curves_a, curves_b] = overlapping_boxes<TestType>();
  auto boolean_curves = tf::cpp::async::make_boolean_with_curves(
      counting_resolver{submissions}, curves_a.mesh(), curves_b.mesh(),
      tf::boolean_op::intersection);
  CHECK(submissions->load(std::memory_order_relaxed) == 2);
  CHECK(boolean_curves.get().mesh.size() > 0);
}

TEMPLATE_TEST_CASE("typed cut facades cover index and connectivity matrices",
                   "[cpp][csg][typed]", float, double) {
  constexpr auto mixed = tf::dynamic_size;
  check_typed_pair<std::int32_t, TestType, std::int32_t, 3, 3>();
  check_typed_pair<std::int32_t, TestType, std::int32_t, 3, mixed>();
  check_typed_pair<std::int32_t, TestType, std::int32_t, mixed, 3>();
  check_typed_pair<std::int32_t, TestType, std::int32_t, mixed, mixed>();
  check_typed_pair<std::int64_t, TestType, std::int64_t, 3, 3>();
  check_typed_pair<std::int64_t, TestType, std::int64_t, 3, mixed>();
  check_typed_pair<std::int64_t, TestType, std::int64_t, mixed, 3>();
  check_typed_pair<std::int64_t, TestType, std::int64_t, mixed, mixed>();
  check_typed_pair<std::int32_t, TestType, std::int64_t>();
  check_typed_pair<std::int64_t, TestType, std::int32_t>();

  static_assert(
      std::is_same_v<
          tf::cpp::boolean_result<std::int32_t, TestType, 3>,
          tf::cpp::boolean_result<tf::cpp::default_index_t, TestType, 3>>);
  static_assert(std::is_same_v<
                tf::cpp::boolean_with_curves_result<std::int32_t, TestType, 3>,
                tf::cpp::boolean_with_curves_result<tf::cpp::default_index_t,
                                                    TestType, 3>>);
}

TEMPLATE_TEST_CASE("typed boolean exact box fixtures cover set relations",
                   "[cpp][csg][typed][boolean]", float, double) {
  const auto owned_outer = typed_box<std::int32_t, TestType>(TestType{4});
  const auto owned_inner =
      typed_box<std::int64_t, TestType, tf::dynamic_size>(TestType{2});
  const auto outer = owned_outer.mesh();
  const auto inner = owned_inner.mesh();

  const auto merged =
      tf::cpp::make_boolean<std::int32_t, TestType, std::int64_t>(
          outer, inner, tf::boolean_op::merge);
  CHECK(merged.mesh.size() == 12);
  CHECK(merged.mesh.points_buffer().size() == 8);
  CHECK(boolean_metadata_in_range(merged, 12, 12));
  for (const auto source : merged.labels.make_range())
    CHECK(source == 0);

  const auto intersection =
      tf::cpp::make_boolean<std::int32_t, TestType, std::int64_t>(
          outer, inner, tf::boolean_op::intersection);
  CHECK(intersection.mesh.size() == 12);
  CHECK(intersection.mesh.points_buffer().size() == 8);
  for (const auto source : intersection.labels.make_range())
    CHECK(source == 1);

  const auto shell =
      tf::cpp::make_boolean<std::int32_t, TestType, std::int64_t>(
          outer, inner, tf::boolean_op::left_difference);
  CHECK(shell.mesh.size() == 24);
  CHECK(shell.mesh.points_buffer().size() == 16);

  const auto impossible =
      tf::cpp::make_boolean<std::int32_t, TestType, std::int64_t>(
          outer, inner, tf::boolean_op::right_difference);
  CHECK(impossible.mesh.size() == 0);
  CHECK(impossible.mesh.points_buffer().size() == 0);

  const auto nested_curves =
      tf::cpp::make_boolean_with_curves<std::int32_t, TestType, std::int64_t>(
          outer, inner, tf::boolean_op::merge);
  CHECK(nested_curves.curves.size() == 0);

  auto owned_distant = typed_box<std::int64_t, TestType>();
  owned_distant.place(translation<TestType>(TestType{10}));
  const auto distant = owned_distant.mesh();
  const auto disjoint_union =
      tf::cpp::make_boolean<std::int32_t, TestType, std::int64_t>(
          outer, distant, tf::boolean_op::merge);
  const auto disjoint_intersection =
      tf::cpp::make_boolean<std::int32_t, TestType, std::int64_t>(
          outer, distant, tf::boolean_op::intersection);
  CHECK(disjoint_union.mesh.size() == 24);
  CHECK(disjoint_union.mesh.points_buffer().size() == 16);
  CHECK(disjoint_intersection.mesh.size() == 0);

  const auto empty =
      typed_empty_mesh<std::int64_t, TestType, tf::dynamic_size>();
  const auto empty_merge =
      tf::cpp::make_boolean<std::int32_t, TestType, std::int64_t>(
          outer, empty.mesh(), tf::boolean_op::merge);
  CHECK(empty_merge.mesh.size() == owned_outer.polygons.size());
  CHECK(empty_merge.mesh.points_buffer().size() ==
        owned_outer.polygons.points_buffer().size());
}

TEMPLATE_TEST_CASE("typed cut validation and async calls state the borrow",
                   "[cpp][csg][typed][async]", float, double) {
  // a face naming a point the geometry does not have is refused at the one
  // door the reading passes, whatever the index width
  const auto peer = typed_box<std::int32_t, TestType, tf::dynamic_size>();
  const boolean_owned<std::int64_t, TestType> malformed{
      tf::cpp::test::polygons_of<std::int64_t, TestType>(
          {0, 1, 2}, {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                      TestType{0}, TestType{0}})};
  CHECK_THROWS_AS((tf::cpp::make_boolean<std::int64_t, TestType, std::int32_t>(
                      malformed.mesh(), peer.mesh(), tf::boolean_op::merge)),
                  std::out_of_range);

  const boolean_owned<std::int64_t, TestType> too_wide{
      tf::cpp::test::polygons_of<std::int64_t, TestType>(
          {static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) +
               1,
           1, 2},
          {TestType{0}, TestType{0}, TestType{0}, TestType{1}, TestType{0},
           TestType{0}, TestType{0}, TestType{1}, TestType{0}})};
  CHECK_THROWS_AS((tf::cpp::make_boolean<std::int64_t, TestType, std::int32_t>(
                      too_wide.mesh(), peer.mesh(), tf::boolean_op::merge)),
                  std::out_of_range);

  auto submissions = std::make_shared<std::atomic<int>>(0);
  const auto boolean_a = typed_box<std::int64_t, TestType, tf::dynamic_size>();
  auto boolean_b = typed_box<std::int32_t, TestType>();
  auto retained_placement = translation<TestType>(TestType{1});
  boolean_b.place(retained_placement);
  auto boolean = tf::cpp::async::make_boolean(
      mutating_resolver{submissions,
                        [&] {
                          // the placement was copied into the fixture's
                          // own slot
                          retained_placement[3] = TestType{100};
                        }},
      boolean_a.mesh(), boolean_b.mesh(), tf::boolean_op::merge);
  static_assert(
      std::is_same_v<
          decltype(boolean),
          tf::cpp::async::resolver_result_t<
              mutating_resolver, tf::cpp::boolean_result<std::int64_t, TestType,
                                                         tf::dynamic_size>>>);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(boolean.get().mesh.size() > 0);
}
