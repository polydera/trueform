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
#include "mixed_mesh.hpp"
#include "parity_checks.hpp"

#include "trueform/cpp/iso/async/isocontours.hpp"
#include "trueform/cpp/iso/isobands.hpp"
#include "trueform/cpp/iso/isocontours.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace tf::cpp::test;

namespace {

/// The coordinates a mesh holds, copied out: a check that asks whether an
/// entry wrote back to its operand needs them as they were.
template <typename Index, typename Real, std::size_t Ngon>
auto point_storage_of(const tf::polygons_buffer<Index, Real, 3, Ngon> &value)
    -> std::vector<Real> {
  const auto &storage = value.points_buffer().data_buffer();
  return {storage.begin(), storage.end()};
}

template <typename Real> auto scalar_square() -> parity_operand<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 2, 3},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{1}, Real{1},
       Real{0}, Real{0}, Real{1}, Real{0}})};
}

template <typename Index, typename Real> struct isocontours_row {
  using real_type = Real;
  using index_type = Index;
  using mesh_type = tf::cpp::test::owned_mesh<Index, Real, 3>;
  using curves_type = tf::curves_buffer<Index, Real, 3>;
};

using isocontours_float_int32 = isocontours_row<std::int32_t, float>;
using isocontours_float_int64 = isocontours_row<std::int64_t, float>;
using isocontours_double_int32 = isocontours_row<std::int32_t, double>;
using isocontours_double_int64 = isocontours_row<std::int64_t, double>;

template <typename Mesh, typename Real, typename = void>
struct has_isocontours : std::false_type {};
template <typename Mesh, typename Real>
struct has_isocontours<Mesh, Real,
                       std::void_t<decltype(tf::cpp::isocontours(
                           std::declval<const Mesh &>(),
                           std::declval<const tf::cpp::nd_array<Real> &>(),
                           std::declval<Real>()))>> : std::true_type {};

static_assert(
    has_isocontours<tf::cpp::mesh<std::int64_t, float, 3, 3>, float>::value);
static_assert(
    !has_isocontours<tf::cpp::mesh<std::int64_t, float, 2, 3>, float>::value);

template <typename Row, std::size_t Ngon>
auto typed_scalar_square()
    -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const std::initializer_list<Real> points{Real{0}, Real{0}, Real{0}, Real{1},
                                           Real{0}, Real{0}, Real{1}, Real{1},
                                           Real{0}, Real{0}, Real{1}, Real{0}};
  if constexpr (Ngon == 3)
    return {
        tf::cpp::test::polygons_of<Index, Real>({0, 1, 2, 0, 2, 3}, points)};
  else
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 3, 6}, {0, 1, 2, 0, 2, 3}, points)};
}

template <typename Row, std::size_t Ngon>
auto typed_empty_mesh()
    -> tf::cpp::test::mesh_at<typename Row::mesh_type, Ngon> {
  return {};
}

template <typename Curves0, typename Curves1>
auto exact_curves_geometry_equal(const Curves0 &first, const Curves1 &second)
    -> bool {
  if (first.size() != second.size())
    return false;
  auto gather = [](const auto &curves) {
    using Real = std::decay_t<decltype(curves.points()[0][0])>;
    std::vector<std::array<Real, 3>> points;
    for (const auto point : curves.points())
      points.push_back({point[0], point[1], point[2]});
    std::sort(points.begin(), points.end());
    return points;
  };
  return gather(first) == gather(second);
}

template <typename Curves, typename Real>
auto exact_vertical_cuts(const Curves &curves, std::vector<Real> expected_cuts)
    -> bool {
  if (!curves_are_aligned(curves, true) ||
      curves.size() != expected_cuts.size())
    return false;

  std::vector<Real> actual_cuts;
  std::size_t iterated_paths = 0;
  for (const auto path : curves.paths()) {
    ++iterated_paths;
    if (path.size() < 2)
      return false;
  }

  std::size_t iterated_curves = 0;
  for (const auto curve : curves.curves()) {
    ++iterated_curves;
    if (curve.size() < 2)
      return false;
    const auto cut = curve[0][0];
    auto min_y = Real{1};
    auto max_y = Real{0};
    for (const auto point : curve) {
      if (point[0] != cut || point[2] != Real{0})
        return false;
      min_y = std::min(min_y, point[1]);
      max_y = std::max(max_y, point[1]);
    }
    if (min_y != Real{0} || max_y != Real{1})
      return false;
    actual_cuts.push_back(cut);
  }
  std::sort(actual_cuts.begin(), actual_cuts.end());
  std::sort(expected_cuts.begin(), expected_cuts.end());
  return iterated_paths == static_cast<std::size_t>(curves.size()) &&
         iterated_curves == static_cast<std::size_t>(curves.size()) &&
         actual_cuts == expected_cuts;
}

struct isocontours_mutating_resolver {
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

template <typename Row, std::size_t Ngon>
auto check_isocontours_case(
    const tf::cpp::nd_array<typename Row::real_type> &scalars,
    const tf::cpp::nd_array<typename Row::real_type> &one_cut,
    const tf::cpp::nd_array<typename Row::real_type> &cuts) -> void {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  using Curves = typename Row::curves_type;
  auto owner = typed_scalar_square<Row, Ngon>();
  const auto original_indices = tf::cpp::test::face_indices_of(owner.polygons);
  const auto original_points = point_storage_of(owner.polygons);
  const auto value = owner.mesh();

  auto scalar = tf::cpp::isocontours(value, scalars, Real{0.5});
  auto array = tf::cpp::isocontours(value, scalars, one_cut);
  auto multiple = tf::cpp::isocontours(value, scalars, cuts);
  STATIC_REQUIRE(std::is_same_v<decltype(scalar), Curves>);
  STATIC_REQUIRE(
      std::is_same_v<
          std::decay_t<decltype(tf::cpp::test::arrays_of(scalar).ids[0])>,
          Index>);
  CHECK(exact_vertical_cuts(scalar, std::vector<Real>{Real{0.5}}));
  CHECK(exact_vertical_cuts(array, std::vector<Real>{Real{0.5}}));
  CHECK(exact_curves_geometry_equal(scalar, array));
  CHECK(exact_vertical_cuts(
      multiple, std::vector<Real>{Real{0.25}, Real{0.5}, Real{0.75}}));

  // a contour is the operand's own, so it is read where it was authored
  owner.place({Real{1}, Real{0}, Real{0}, Real{7}, Real{0}, Real{1}, Real{0},
               Real{-2}, Real{0}, Real{0}, Real{1}, Real{4}, Real{0}, Real{0},
               Real{0}, Real{1}});
  const auto transformed = tf::cpp::isocontours(owner.mesh(), scalars, cuts);
  CHECK(exact_vertical_cuts(
      transformed, std::vector<Real>{Real{0.25}, Real{0.5}, Real{0.75}}));
  CHECK(owner.placement.placed);
  CHECK(point_storage_of(owner.polygons) == original_points);
  CHECK(same_array(tf::cpp::test::face_indices_of(owner.polygons),
                   original_indices));
}

} // namespace
TEMPLATE_TEST_CASE(
    "Python-parity isobands align bands curves provenance and local frame",
    "[cpp][iso][python-parity][isobands]", float, double) {
  auto owner = octahedron<TestType>();
  const auto mesh = owner.mesh();
  auto scalars = tf::cpp::test::make_nd_array<TestType>(
      {TestType{-1}, TestType{-0.2}, TestType{-0.1}, TestType{0.1},
       TestType{0.2}, TestType{1}},
      {6});
  auto cuts = tf::cpp::test::make_nd_array<TestType>(
      {TestType{-0.5}, TestType{0}, TestType{0.5}}, {3});
  auto selected = tf::cpp::test::make_nd_array<std::int32_t>({1, 2}, {2});
  const auto original_faces = tf::cpp::test::face_indices_of(owner.polygons);
  const auto original_points = point_storage_of(owner.polygons);
  const auto original_scalars = scalars.deep_copy();
  const auto original_cuts = cuts.deep_copy();
  const auto original_selected = selected.deep_copy();

  const auto all = tf::cpp::isobands_with_curves(mesh, scalars, cuts);
  REQUIRE(mesh_indices_are_valid(all.mesh));
  REQUIRE(curves_are_aligned(all.curves, true));
  REQUIRE(all.labels.length() == all.mesh.size());
  REQUIRE(all.face_labels.length() == all.labels.length());
  REQUIRE(isoband_provenance_is_valid(all, owner, scalars, cuts));
  std::set<std::int32_t> labels;
  for (std::size_t face = 0; face < all.labels.length(); ++face) {
    CHECK(all.face_labels[face] >= 0);
    CHECK(static_cast<std::size_t>(all.face_labels[face]) <
          owner.polygons.size());
    labels.insert(all.labels[face]);
  }
  CHECK(labels == std::set<std::int32_t>{0, 1, 2, 3});
  CHECK(all.curves.size() == 3);

  const auto middle =
      tf::cpp::isobands_with_curves_selected(mesh, scalars, cuts, selected);
  REQUIRE(mesh_indices_are_valid(middle.mesh));
  REQUIRE(curves_are_aligned(middle.curves, true));
  REQUIRE(middle.labels.length() == middle.mesh.size());
  REQUIRE(isoband_provenance_is_valid(middle, owner, scalars, cuts));
  std::set<std::int32_t> middle_labels;
  for (const auto label : middle.labels.make_range())
    middle_labels.insert(label);
  CHECK(middle_labels == std::set<std::int32_t>{1, 2});

  const auto repeated = tf::cpp::isobands_with_curves(mesh, scalars, cuts);
  CHECK(tf::cpp::test::canonicalize_mesh_geometry(
            all.mesh, tf::cpp::test::orientation_mode::preserve) ==
        tf::cpp::test::canonicalize_mesh_geometry(
            repeated.mesh, tf::cpp::test::orientation_mode::preserve));
  CHECK(tf::cpp::test::canonicalize_curves_geometry(
            all.curves, tf::cpp::test::orientation_mode::ignore, true) ==
        tf::cpp::test::canonicalize_curves_geometry(
            repeated.curves, tf::cpp::test::orientation_mode::ignore, true));
  CHECK(
      canonical_face_provenance(
          all.mesh,
          [&](int face) { return all.labels[static_cast<std::size_t>(face)]; },
          [&](int face) {
            return all.face_labels[static_cast<std::size_t>(face)];
          }) ==
      canonical_face_provenance(
          repeated.mesh,
          [&](int face) {
            return repeated.labels[static_cast<std::size_t>(face)];
          },
          [&](int face) {
            return repeated.face_labels[static_cast<std::size_t>(face)];
          }));

  // a band is the operand's own, so it is read where the operand was authored
  owner.place({TestType{1}, TestType{0}, TestType{0}, TestType{5}, TestType{0},
               TestType{1}, TestType{0}, TestType{-3}, TestType{0}, TestType{0},
               TestType{1}, TestType{2}, TestType{0}, TestType{0}, TestType{0},
               TestType{1}});
  const auto transformed =
      tf::cpp::isobands_with_curves(owner.mesh(), scalars, cuts);
  REQUIRE(isoband_provenance_is_valid(transformed, owner, scalars, cuts));
  CHECK(tf::cpp::test::canonicalize_mesh_geometry(
            all.mesh, tf::cpp::test::orientation_mode::preserve) ==
        tf::cpp::test::canonicalize_mesh_geometry(
            transformed.mesh, tf::cpp::test::orientation_mode::preserve));
  CHECK(tf::cpp::test::canonicalize_curves_geometry(
            all.curves, tf::cpp::test::orientation_mode::ignore, true) ==
        tf::cpp::test::canonicalize_curves_geometry(
            transformed.curves, tf::cpp::test::orientation_mode::ignore, true));

  CHECK(same_array(tf::cpp::test::face_indices_of(owner.polygons),
                   original_faces));
  CHECK(point_storage_of(owner.polygons) == original_points);
  CHECK(same_array(scalars, original_scalars));
  CHECK(same_array(cuts, original_cuts));
  CHECK(same_array(selected, original_selected));

  const auto matrix_scalars = tf::cpp::test::make_nd_array<TestType>(
      {TestType{-1}, TestType{0}, TestType{0}, TestType{0}, TestType{0},
       TestType{1}},
      {1, 6});
  CHECK_THROWS_AS(tf::cpp::isobands(mesh, matrix_scalars, cuts),
                  std::invalid_argument);
  const auto invalid_selected =
      tf::cpp::test::make_nd_array<std::int32_t>({4}, {1});
  CHECK_THROWS_AS(
      tf::cpp::isobands_selected(mesh, scalars, cuts, invalid_selected),
      std::out_of_range);
}

TEMPLATE_TEST_CASE("Python-parity isocontours align offsets thresholds and "
                   "scalar presentation",
                   "[cpp][iso][python-parity][isocontours]", float, double) {
  auto owner = scalar_square<TestType>();
  const auto mesh = owner.mesh();
  auto scalars = tf::cpp::test::make_nd_array<TestType>(
      {TestType{0}, TestType{1}, TestType{1}, TestType{0}}, {4});
  auto one_cut = tf::cpp::test::make_nd_array<TestType>({TestType{0.5}}, {1});
  auto cuts = tf::cpp::test::make_nd_array<TestType>(
      {TestType{0.25}, TestType{0.5}, TestType{0.75}}, {3});
  const auto original_faces = tf::cpp::test::face_indices_of(owner.polygons);
  const auto original_points = point_storage_of(owner.polygons);
  const auto original_scalars = scalars.deep_copy();
  const auto original_cuts = cuts.deep_copy();

  const auto scalar = tf::cpp::isocontours(mesh, scalars, TestType{0.5});
  const auto array = tf::cpp::isocontours(mesh, scalars, one_cut);
  const auto multiple = tf::cpp::isocontours(mesh, scalars, cuts);
  REQUIRE(curves_are_aligned(scalar, true));
  REQUIRE(curves_are_aligned(array, true));
  REQUIRE(curves_are_aligned(multiple, true));
  CHECK(tf::cpp::test::canonicalize_curves_geometry(
            scalar, tf::cpp::test::orientation_mode::ignore, true) ==
        tf::cpp::test::canonicalize_curves_geometry(
            array, tf::cpp::test::orientation_mode::ignore, true));
  CHECK(multiple.size() == 3);

  const auto multiple_points = tf::cpp::test::arrays_of(multiple).points;
  for (std::size_t point = 0; point < multiple_points.length(); point += 3) {
    const auto x = multiple_points[point];
    const auto matches_cut =
        tf::cpp::test::within_tolerance(x, TestType{0.25}) ||
        tf::cpp::test::within_tolerance(x, TestType{0.5}) ||
        tf::cpp::test::within_tolerance(x, TestType{0.75});
    CHECK(matches_cut);
    CHECK(tf::cpp::test::within_tolerance(multiple_points[point + 2],
                                          TestType{0}));
  }

  // a contour is the operand's own, so it is read where it was authored
  owner.place({TestType{1}, TestType{0}, TestType{0}, TestType{7}, TestType{0},
               TestType{1}, TestType{0}, TestType{-2}, TestType{0}, TestType{0},
               TestType{1}, TestType{4}, TestType{0}, TestType{0}, TestType{0},
               TestType{1}});
  const auto transformed = tf::cpp::isocontours(owner.mesh(), scalars, cuts);
  CHECK(tf::cpp::test::canonicalize_curves_geometry(
            multiple, tf::cpp::test::orientation_mode::ignore, true) ==
        tf::cpp::test::canonicalize_curves_geometry(
            transformed, tf::cpp::test::orientation_mode::ignore, true));

  CHECK(same_array(tf::cpp::test::face_indices_of(owner.polygons),
                   original_faces));
  CHECK(point_storage_of(owner.polygons) == original_points);
  CHECK(same_array(scalars, original_scalars));
  CHECK(same_array(cuts, original_cuts));

  const auto matrix_scalars = tf::cpp::test::make_nd_array<TestType>(
      {TestType{0}, TestType{1}, TestType{1}, TestType{0}}, {1, 4});
  const auto matrix_cuts = tf::cpp::test::make_nd_array<TestType>(
      {TestType{0.25}, TestType{0.75}}, {1, 2});
  CHECK_THROWS_AS(tf::cpp::isocontours(mesh, matrix_scalars, TestType{0.5}),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::isocontours(mesh, scalars, matrix_cuts),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "Python-parity typed isocontours cover real index and runtime layouts",
    "[cpp][iso][python-parity][isocontours][matrix][fixed][dynamic]",
    isocontours_float_int32, isocontours_float_int64, isocontours_double_int32,
    isocontours_double_int64) {
  using Real = typename TestType::real_type;

  auto scalars = tf::cpp::test::make_nd_array<Real>(
      {Real{0}, Real{1}, Real{1}, Real{0}}, {4});
  auto one_cut = tf::cpp::test::make_nd_array<Real>({Real{0.5}}, {1});
  auto cuts = tf::cpp::test::make_nd_array<Real>(
      {Real{0.75}, Real{0.25}, Real{0.5}, Real{0.5}}, {4});
  const auto original_scalars = scalars.deep_copy();
  const auto original_cuts = cuts.deep_copy();

  check_isocontours_case<TestType, 3>(scalars, one_cut, cuts);
  check_isocontours_case<TestType, tf::dynamic_size>(scalars, one_cut, cuts);

  CHECK(same_array(scalars, original_scalars));
  CHECK(same_array(cuts, original_cuts));
}

TEMPLATE_TEST_CASE(
    "Python-parity typed isocontours validate arrays indices and empty forms",
    "[cpp][iso][python-parity][isocontours][matrix][validation][empty]",
    isocontours_float_int32, isocontours_float_int64, isocontours_double_int32,
    isocontours_double_int64) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Owned = typename TestType::mesh_type;

  const auto owner = typed_scalar_square<TestType, 3>();
  const auto value = owner.mesh();
  auto scalars = tf::cpp::test::make_nd_array<Real>(
      {Real{0}, Real{1}, Real{1}, Real{0}}, {4});
  auto cuts = tf::cpp::test::make_nd_array<Real>({Real{0.25}, Real{0.75}}, {2});
  auto matrix_scalars = tf::cpp::test::make_nd_array<Real>(
      {Real{0}, Real{1}, Real{1}, Real{0}}, {1, 4});
  auto short_scalars =
      tf::cpp::test::make_nd_array<Real>({Real{0}, Real{1}}, {2});
  auto matrix_cuts =
      tf::cpp::test::make_nd_array<Real>({Real{0.25}, Real{0.75}}, {1, 2});
  tf::cpp::nd_array<Real> invalid_array;
  const Owned empty_owner;
  const auto invalid_mesh = empty_owner.mesh();

  CHECK_THROWS_AS(
      (tf::cpp::isocontours<Index, Real>(invalid_mesh, scalars, Real{0.5})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::isocontours<Index, Real>(value, invalid_array, Real{0.5})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::isocontours<Index, Real>(value, matrix_scalars, Real{0.5})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::isocontours<Index, Real>(value, short_scalars, Real{0.5})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::isocontours<Index, Real>(value, scalars, invalid_array)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::isocontours<Index, Real>(value, scalars, matrix_cuts)),
      std::invalid_argument);

  // faces that name a point the geometry does not have are refused at the one
  // door the reading passes, at either arity
  const std::initializer_list<Real> two_points{Real{0}, Real{0}, Real{0},
                                               Real{1}, Real{0}, Real{0}};
  const auto short_field =
      tf::cpp::test::make_nd_array<Real>({Real{0}, Real{1}}, {2});
  const Owned bad_fixed{
      tf::cpp::test::polygons_of<Index, Real>({0, 1, 2}, two_points)};
  CHECK_THROWS_AS(
      (tf::cpp::isocontours(bad_fixed.mesh(), short_field, Real{0.5})),
      std::out_of_range);

  const tf::cpp::test::mixed_mesh_of<Owned> bad_dynamic{
      tf::cpp::test::polygons_of<Index, Real>({0, 3}, {0, 1, 2}, two_points)};
  CHECK_THROWS_AS(
      (tf::cpp::isocontours(bad_dynamic.mesh(), short_field, Real{0.5})),
      std::out_of_range);

  auto no_cuts = tf::cpp::test::make_nd_array<Real>({}, {0});
  const auto no_thresholds =
      tf::cpp::isocontours<Index, Real>(value, scalars, no_cuts);
  CHECK(curves_are_aligned(no_thresholds, false));
  CHECK(no_thresholds.size() == 0);
  CHECK(tf::cpp::test::arrays_of(no_thresholds).points.raw_shape() ==
        tf::small_vector<int, 3>{0, 3});

  const auto refuses_nothing = [&](const auto &empty) {
    auto empty_scalars = tf::cpp::test::make_nd_array<Real>({}, {0});
    const auto result =
        tf::cpp::isocontours(empty.mesh(), empty_scalars, Real{0.5});
    CHECK(curves_are_aligned(result, false));
    CHECK(result.size() == 0);
    CHECK(tf::cpp::test::arrays_of(result).points.raw_shape() ==
          tf::small_vector<int, 3>{0, 3});
  };
  refuses_nothing(typed_empty_mesh<TestType, 3>());
  refuses_nothing(typed_empty_mesh<TestType, tf::dynamic_size>());
}

TEST_CASE("Python-parity typed isocontour async owns its scalar arrays",
          "[cpp][iso][python-parity][isocontours][async][ownership]") {
  using Row = isocontours_double_int64;
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  using Result = typename Row::curves_type;

  auto owner = typed_scalar_square<Row, tf::dynamic_size>();
  auto scalars = tf::cpp::test::make_nd_array<Real>(
      {Real{0}, Real{1}, Real{1}, Real{0}}, {4});
  auto cuts = tf::cpp::test::make_nd_array<Real>(
      {Real{0.25}, Real{0.5}, Real{0.75}}, {3});
  auto retained_placement = std::array<Real, 16>{
      Real{1}, Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0},
      Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{0}, Real{1}};
  owner.place(retained_placement);
  const auto value = owner.mesh();
  const auto expected = tf::cpp::isocontours<Index, Real>(value, scalars, cuts);
  const auto submissions = std::make_shared<std::atomic<int>>(0);

  auto pending = tf::cpp::async::isocontours(
      isocontours_mutating_resolver{submissions,
                                    [&] {
                                      // the placement was copied into the
                                      // fixture's own slot, and the field
                                      // arrays are the job's to keep once it
                                      // was handed them
                                      retained_placement[3] = Real{10};
                                      scalars.destroy();
                                      cuts.destroy();
                                    }},
      value, scalars, cuts);
  STATIC_REQUIRE(std::is_same_v<decltype(pending), std::future<Result>>);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(exact_vertical_cuts(
      expected, std::vector<Real>{Real{0.25}, Real{0.5}, Real{0.75}}));
  CHECK(exact_vertical_cuts(
      pending.get(), std::vector<Real>{Real{0.25}, Real{0.5}, Real{0.75}}));

  const auto legacy = scalar_square<float>();
  auto legacy_scalars = tf::cpp::test::make_nd_array<float>({0, 1, 1, 0}, {4});
  auto legacy_pending = tf::cpp::async::isocontours(
      isocontours_mutating_resolver{submissions,
                                    [&] { legacy_scalars.destroy(); }},
      legacy.mesh(), legacy_scalars, 0.5F);
  STATIC_REQUIRE(
      std::is_same_v<
          decltype(legacy_pending),
          std::future<tf::curves_buffer<tf::cpp::default_index_t, float, 3>>>);
  CHECK(exact_vertical_cuts(legacy_pending.get(), std::vector<float>{0.5F}));
  CHECK(submissions->load(std::memory_order_relaxed) == 2);
}
