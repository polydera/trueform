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
#include "trueform/cpp/iso/async/isobands.hpp"
#include "trueform/cpp/iso/isobands.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename T>
auto array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

/// A result is core's own storage, so a check reads its arrays where they lie.
template <typename Index, typename Real, std::size_t Ngon>
auto result_points(const tf::polygons_buffer<Index, Real, 3, Ngon> &value)
    -> tf::cpp::nd_array<Real> {
  const auto &storage = value.points_buffer().data_buffer();
  const auto count = static_cast<int>(value.points_buffer().size());
  return tf::cpp::nd_array<Real>::from_borrowed(
      {}, const_cast<Real *>(storage.data()), storage.size(), {count, 3});
}

template <typename Real>
using band_owned = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real> auto square() -> band_owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 2, 3},
      {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0}, Real{1},
       Real{1}, Real{0}, Real{-1}, Real{1}, Real{0}})};
}

template <typename Real> auto scalars() -> tf::cpp::nd_array<Real> {
  return array<Real>({-1, 1, 1, -1}, {4});
}

template <typename Real> auto cuts() -> tf::cpp::nd_array<Real> {
  return array<Real>({-0.5, 0, 0.5}, {3});
}

template <typename Real> auto empty_mesh() -> band_owned<Real> { return {}; }

template <typename Real, std::size_t Ngon>
auto coherent(const tf::cpp::isobands_result<tf::cpp::default_index_t, Real,
                                             Ngon> &result) -> bool {
  return result.labels.is_valid() && result.face_labels.is_valid() &&
         result.labels.ndim() == 1 && result.face_labels.ndim() == 1 &&
         result.labels.length() == result.mesh.size() &&
         result.face_labels.length() == result.labels.length();
}

template <typename Real, std::size_t Ngon>
auto coherent(const tf::cpp::isobands_with_curves_result<
              tf::cpp::default_index_t, Real, Ngon> &result) -> bool {
  return coherent(
             tf::cpp::isobands_result<tf::cpp::default_index_t, Real, Ngon>{
                 result.mesh, result.labels, result.face_labels}) &&
         tf::cpp::test::arrays_of(result.curves).points.ndim() == 2 &&
         tf::cpp::test::arrays_of(result.curves).points.shape_at(1) == 3;
}

template <typename T>
auto same_array(const tf::cpp::nd_array<T> &first,
                const tf::cpp::nd_array<T> &second) -> bool {
  if (first.raw_shape() != second.raw_shape() ||
      first.length() != second.length())
    return false;
  for (std::size_t index = 0; index < first.length(); ++index)
    if (first[index] != second[index])
      return false;
  return true;
}

template <typename Real>
auto same_mesh(
    const tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3> &first,
    const tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3> &second)
    -> bool {
  return same_array(tf::cpp::test::face_indices_of(first),
                    tf::cpp::test::face_indices_of(second)) &&
         same_array(result_points(first), result_points(second));
}

template <typename Real>
auto same_curves(
    const tf::curves_buffer<tf::cpp::default_index_t, Real, 3> &first,
    const tf::curves_buffer<tf::cpp::default_index_t, Real, 3> &second)
    -> bool {
  using point_type = std::array<Real, 3>;
  using path_type = std::vector<point_type>;
  auto canonical_path = [](path_type path) {
    const auto closed = path.size() > 1 && path.front() == path.back();
    if (closed)
      path.pop_back();
    if (path.empty())
      return path;

    auto reversed = path;
    std::reverse(reversed.begin(), reversed.end());
    if (!closed)
      return std::min(path, reversed);

    auto best = path;
    auto consider_rotations = [&](const path_type &candidate) {
      for (std::size_t shift = 0; shift < candidate.size(); ++shift) {
        path_type rotated;
        rotated.reserve(candidate.size());
        rotated.insert(rotated.end(), candidate.begin() + shift,
                       candidate.end());
        rotated.insert(rotated.end(), candidate.begin(),
                       candidate.begin() + shift);
        if (rotated < best)
          best = std::move(rotated);
      }
    };
    consider_rotations(path);
    consider_rotations(reversed);
    best.push_back(best.front());
    return best;
  };

  auto signature =
      [&](const tf::curves_buffer<tf::cpp::default_index_t, Real, 3> &value) {
        std::vector<path_type> paths;
        const auto arrays = tf::cpp::test::arrays_of(value);
        const auto &offsets = arrays.offsets;
        const auto &data = arrays.ids;
        const auto &points = arrays.points;
        paths.reserve(offsets.empty() ? 0 : offsets.length() - 1);
        for (std::size_t path_id = 1; path_id < offsets.length(); ++path_id) {
          path_type path;
          path.reserve(static_cast<std::size_t>(offsets[path_id] -
                                                offsets[path_id - 1]));
          for (auto index = offsets[path_id - 1]; index < offsets[path_id];
               ++index) {
            const auto point_offset = static_cast<std::size_t>(data[index]) * 3;
            path.push_back({points[point_offset], points[point_offset + 1],
                            points[point_offset + 2]});
          }
          paths.push_back(canonical_path(std::move(path)));
        }
        std::sort(paths.begin(), paths.end());
        return paths;
      };
  return tf::cpp::test::arrays_of(first).points.shape_at(0) ==
             tf::cpp::test::arrays_of(second).points.shape_at(0) &&
         signature(first) == signature(second);
}

template <typename Real, std::size_t Ngon>
auto same_result(
    const tf::cpp::isobands_result<tf::cpp::default_index_t, Real, Ngon> &first,
    const tf::cpp::isobands_result<tf::cpp::default_index_t, Real, Ngon>
        &second) -> bool {
  return same_mesh(first.mesh, second.mesh) &&
         same_array(first.labels, second.labels) &&
         same_array(first.face_labels, second.face_labels);
}

template <typename Real, std::size_t Ngon>
auto same_result(
    const tf::cpp::isobands_with_curves_result<tf::cpp::default_index_t, Real,
                                               Ngon> &first,
    const tf::cpp::isobands_with_curves_result<tf::cpp::default_index_t, Real,
                                               Ngon> &second) -> bool {
  return same_mesh(first.mesh, second.mesh) &&
         same_array(first.labels, second.labels) &&
         same_array(first.face_labels, second.face_labels) &&
         same_curves(first.curves, second.curves);
}

struct mutating_resolver {
  std::shared_ptr<std::atomic<int>> submissions;
  std::function<void()> mutate;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    mutate();
    return std::make_shared<state_type<T>>();
  }
};

template <typename Index, typename Real, std::size_t Ngon = 3>
auto python_octahedron() -> tf::cpp::test::owned_mesh<Index, Real, 3, Ngon> {
  const std::initializer_list<Index> faces{0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1,
                                           5, 2, 1, 5, 3, 2, 5, 4, 3, 5, 1, 4};
  const std::initializer_list<Real> points{
      Real{0}, Real{0},  Real{-1}, Real{1},  Real{0}, Real{0},
      Real{0}, Real{1},  Real{0},  Real{-1}, Real{0}, Real{0},
      Real{0}, Real{-1}, Real{0},  Real{0},  Real{0}, Real{1}};
  if constexpr (Ngon == 3)
    return {tf::cpp::test::polygons_of<Index, Real>(faces, points)};
  else
    return {tf::cpp::test::polygons_of<Index, Real>(
        {0, 3, 6, 9, 12, 15, 18, 21, 24}, faces, points)};
}

template <typename Index, typename Real, std::size_t Ngon = 3>
auto typed_empty_mesh() -> tf::cpp::test::owned_mesh<Index, Real, 3, Ngon> {
  return {};
}

template <typename Index, typename Real>
auto typed_curves_aligned(const tf::curves_buffer<Index, Real, 3> &value)
    -> bool {
  const auto arrays = tf::cpp::test::arrays_of(value);
  const auto &offsets = arrays.offsets;
  const auto &data = arrays.ids;
  const auto &points = arrays.points;
  if (points.ndim() != 2 || points.shape_at(1) != 3)
    return false;
  if (value.size() == 0)
    return data.empty() &&
           (offsets.empty() || (offsets.length() == 1 && offsets[0] == 0));
  if (offsets.empty() || offsets[0] != Index{0} ||
      offsets.length() != static_cast<std::size_t>(value.size() + 1) ||
      offsets[offsets.length() - 1] != static_cast<Index>(data.length()))
    return false;
  for (std::size_t path = 1; path < offsets.length(); ++path)
    if (offsets[path] < offsets[path - 1])
      return false;
  for (const auto point : data)
    if (point < Index{0} || point >= static_cast<Index>(points.shape_at(0)))
      return false;
  return true;
}

template <typename Index, typename Real>
auto typed_curve_signature(const tf::curves_buffer<Index, Real, 3> &value) {
  using point_type = std::array<Real, 3>;
  using path_type = std::vector<point_type>;
  std::vector<path_type> paths;
  const auto arrays = tf::cpp::test::arrays_of(value);
  const auto &offsets = arrays.offsets;
  const auto &data = arrays.ids;
  const auto &coordinates = arrays.points;
  paths.reserve(offsets.empty() ? 0 : offsets.length() - 1);
  for (std::size_t path_id = 1; path_id < offsets.length(); ++path_id) {
    path_type path;
    for (auto at = offsets[path_id - 1]; at < offsets[path_id]; ++at) {
      const auto point = static_cast<std::size_t>(data[at]) * 3;
      path.push_back(
          {coordinates[point], coordinates[point + 1], coordinates[point + 2]});
    }
    const auto closed = path.size() > 1 && path.front() == path.back();
    if (closed)
      path.pop_back();
    if (!path.empty()) {
      auto reversed = path;
      std::reverse(reversed.begin(), reversed.end());
      if (!closed) {
        path = std::min(path, reversed);
      } else {
        auto best = path;
        auto consider = [&](const path_type &candidate) {
          for (std::size_t shift = 0; shift < candidate.size(); ++shift) {
            path_type rotated;
            rotated.insert(rotated.end(), candidate.begin() + shift,
                           candidate.end());
            rotated.insert(rotated.end(), candidate.begin(),
                           candidate.begin() + shift);
            best = std::min(best, rotated);
          }
        };
        consider(path);
        consider(reversed);
        path = std::move(best);
        path.push_back(path.front());
      }
    }
    paths.push_back(std::move(path));
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

template <typename Index, typename Real, std::size_t Ngon>
auto same_typed_mesh(const tf::polygons_buffer<Index, Real, 3, Ngon> &first,
                     const tf::polygons_buffer<Index, Real, 3, Ngon> &second)
    -> bool {
  return same_array(tf::cpp::test::face_indices_of(first),
                    tf::cpp::test::face_indices_of(second)) &&
         same_array(result_points(first), result_points(second));
}

template <typename Index, typename Real, std::size_t Ngon>
auto same_typed_result(
    const tf::cpp::isobands_with_curves_result<Index, Real, Ngon> &first,
    const tf::cpp::isobands_with_curves_result<Index, Real, Ngon> &second)
    -> bool {
  return same_typed_mesh(first.mesh, second.mesh) &&
         same_array(first.labels, second.labels) &&
         same_array(first.face_labels, second.face_labels) &&
         typed_curve_signature<Index, Real>(first.curves) ==
             typed_curve_signature<Index, Real>(second.curves);
}

} // namespace

TEMPLATE_TEST_CASE("isobands returns every typed result category",
                   "[cpp][iso][isobands][sync]", float, double) {
  const auto owner = square<TestType>();
  const auto mesh = owner.mesh();
  const auto scalar_field = scalars<TestType>();
  const auto cut_values = cuts<TestType>();
  const auto selected = array<std::int32_t>({1}, {1});

  const auto all = tf::cpp::isobands(mesh, scalar_field, cut_values);
  const auto all_curves =
      tf::cpp::isobands_with_curves(mesh, scalar_field, cut_values);
  const auto one =
      tf::cpp::isobands_selected(mesh, scalar_field, cut_values, selected);
  const auto one_curves = tf::cpp::isobands_with_curves_selected(
      mesh, scalar_field, cut_values, selected);

  static_assert(
      std::is_same_v<decltype(all),
                     const tf::cpp::isobands_result<tf::cpp::default_index_t,
                                                    TestType, 3>>);
  static_assert(std::is_same_v<decltype(all_curves),
                               const tf::cpp::isobands_with_curves_result<
                                   tf::cpp::default_index_t, TestType, 3>>);
  CHECK(coherent(all));
  CHECK(coherent(all_curves));
  CHECK(coherent(one));
  CHECK(coherent(one_curves));
  CHECK(all.mesh.size() >= owner.polygons.size());
  CHECK(all_curves.curves.size() > 0);
  CHECK(one.mesh.size() > 0);
  for (const auto label : one.labels)
    CHECK(label == 1);
}

TEMPLATE_TEST_CASE("isobands preserves empty inputs and owned outputs",
                   "[cpp][iso][isobands][empty][ownership]", float, double) {
  const auto owner = square<TestType>();
  auto scalar_field = scalars<TestType>();
  auto no_cuts = array<TestType>({}, {0});
  auto none = array<std::int32_t>({}, {0});

  auto uncut = tf::cpp::isobands(owner.mesh(), scalar_field, no_cuts);
  auto empty_selection =
      tf::cpp::isobands_selected(owner.mesh(), scalar_field, no_cuts, none);
  CHECK(coherent(uncut));
  CHECK(uncut.mesh.size() == owner.polygons.size());
  for (const auto label : uncut.labels)
    CHECK(label == 0);
  CHECK(coherent(empty_selection));
  CHECK(empty_selection.mesh.size() == 0);

  const auto empty = empty_mesh<TestType>();
  auto empty_scalars = array<TestType>({}, {0});
  auto empty_result = tf::cpp::isobands_with_curves(empty.mesh(), empty_scalars,
                                                    array<TestType>({0}, {1}));
  CHECK(coherent(empty_result));
  CHECK(empty_result.mesh.size() == 0);
  CHECK(empty_result.curves.size() == 0);

  // the result owns its own storage, so the inputs it was read from may go
  scalar_field.destroy();
  no_cuts.destroy();
  none.destroy();
  CHECK(uncut.labels.is_valid());
  CHECK(uncut.face_labels.is_valid());
}

TEMPLATE_TEST_CASE("isobands validates mesh scalar cuts and selected bands",
                   "[cpp][iso][isobands][validation]", float, double) {
  const auto owner = square<TestType>();
  const auto mesh = owner.mesh();
  const auto scalar_field = scalars<TestType>();
  const auto cut_values = cuts<TestType>();
  const band_owned<TestType> empty_owner;
  const auto invalid_mesh = empty_owner.mesh();

  CHECK_THROWS_AS(tf::cpp::isobands(invalid_mesh, scalar_field, cut_values),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::isobands(mesh, array<TestType>({0, 1, 2}, {3}), cut_values),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::isobands(mesh, scalar_field, array<TestType>({0, 1}, {1, 2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::isobands(
          mesh,
          array<TestType>(
              {-1, 1, 1, std::numeric_limits<TestType>::quiet_NaN()}, {4}),
          cut_values),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::isobands(
          mesh, scalar_field,
          array<TestType>({std::numeric_limits<TestType>::infinity()}, {1})),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::isobands_selected(mesh, scalar_field, cut_values,
                                             array<std::int32_t>({1}, {1, 1})),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::isobands_selected(mesh, scalar_field, cut_values,
                                             array<std::int32_t>({-1}, {1})),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::isobands_selected(mesh, scalar_field, cut_values,
                                             array<std::int32_t>({4}, {1})),
                  std::out_of_range);

  // faces that name a point the geometry does not have are refused at the one
  // door the reading passes
  const band_owned<TestType> shrunk{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                      TestType{0}, TestType{0}})};
  CHECK_THROWS_AS(tf::cpp::isobands(shrunk.mesh(), array<TestType>({0, 1}, {2}),
                                    cut_values),
                  std::out_of_range);
}

TEMPLATE_TEST_CASE("isobands rejects non-finite mesh coordinates everywhere",
                   "[cpp][iso][isobands][review][finite-coordinates]", float,
                   double) {
  const auto selected = array<std::int32_t>({1}, {1});
  for (const auto coordinate : {std::numeric_limits<TestType>::infinity(),
                                std::numeric_limits<TestType>::quiet_NaN()}) {
    auto owner = square<TestType>();
    const auto scalar_field = scalars<TestType>();
    const auto cut_values = cuts<TestType>();
    owner.polygons.points_buffer().data_buffer()[0] = coordinate;
    owner.cache.points_changed();
    const auto mesh = owner.mesh();

    CHECK_THROWS_AS(tf::cpp::isobands(mesh, scalar_field, cut_values),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        tf::cpp::isobands_with_curves(mesh, scalar_field, cut_values),
        std::invalid_argument);
    CHECK_THROWS_AS(
        tf::cpp::isobands_selected(mesh, scalar_field, cut_values, selected),
        std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::isobands_with_curves_selected(
                        mesh, scalar_field, cut_values, selected),
                    std::invalid_argument);

    auto all = tf::cpp::async::isobands(mesh, scalar_field, cut_values);
    auto all_curves =
        tf::cpp::async::isobands_with_curves(mesh, scalar_field, cut_values);
    auto one = tf::cpp::async::isobands_selected(mesh, scalar_field, cut_values,
                                                 selected);
    auto one_curves = tf::cpp::async::isobands_with_curves_selected(
        mesh, scalar_field, cut_values, selected);
    CHECK_THROWS_AS(all.get(), std::invalid_argument);
    CHECK_THROWS_AS(all_curves.get(), std::invalid_argument);
    CHECK_THROWS_AS(one.get(), std::invalid_argument);
    CHECK_THROWS_AS(one_curves.get(), std::invalid_argument);
  }
}

TEMPLATE_TEST_CASE("isobands async retains the field arrays it was handed",
                   "[cpp][iso][isobands][async][review][ownership]", float,
                   double) {
  // THE ASYNC ARRAY LAW: a job carries the HANDLE, so the caller may release
  // its own after dispatch and the storage stays.
  auto submissions = std::make_shared<std::atomic<int>>(0);

  {
    const auto owner = square<TestType>();
    auto scalar_field = scalars<TestType>();
    auto cut_values = cuts<TestType>();
    const auto baseline =
        tf::cpp::isobands(owner.mesh(), scalar_field, cut_values);
    auto future =
        tf::cpp::async::isobands(mutating_resolver{submissions,
                                                   [&] {
                                                     scalar_field.destroy();
                                                     cut_values.destroy();
                                                   }},
                                 owner.mesh(), scalar_field, cut_values);
    CHECK(same_result(baseline, future.get()));
  }
  {
    const auto owner = square<TestType>();
    auto scalar_field = scalars<TestType>();
    auto cut_values = cuts<TestType>();
    const auto baseline =
        tf::cpp::isobands_with_curves(owner.mesh(), scalar_field, cut_values);
    auto future = tf::cpp::async::isobands_with_curves(
        mutating_resolver{submissions,
                          [&] {
                            scalar_field.destroy();
                            cut_values.destroy();
                          }},
        owner.mesh(), scalar_field, cut_values);
    CHECK(same_result(baseline, future.get()));
  }
  {
    const auto owner = square<TestType>();
    auto scalar_field = scalars<TestType>();
    auto cut_values = cuts<TestType>();
    auto selected = array<std::int32_t>({1}, {1});
    const auto baseline = tf::cpp::isobands_selected(owner.mesh(), scalar_field,
                                                     cut_values, selected);
    auto future = tf::cpp::async::isobands_selected(
        mutating_resolver{submissions,
                          [&] {
                            scalar_field.destroy();
                            cut_values.destroy();
                            selected.destroy();
                          }},
        owner.mesh(), scalar_field, cut_values, selected);
    CHECK(same_result(baseline, future.get()));
  }
  {
    const auto owner = square<TestType>();
    auto scalar_field = scalars<TestType>();
    auto cut_values = cuts<TestType>();
    auto selected = array<std::int32_t>({1}, {1});
    const auto baseline = tf::cpp::isobands_with_curves_selected(
        owner.mesh(), scalar_field, cut_values, selected);
    auto future = tf::cpp::async::isobands_with_curves_selected(
        mutating_resolver{submissions,
                          [&] {
                            scalar_field.destroy();
                            cut_values.destroy();
                            selected.destroy();
                          }},
        owner.mesh(), scalar_field, cut_values, selected);
    CHECK(same_result(baseline, future.get()));
  }
  CHECK(submissions->load(std::memory_order_relaxed) == 4);

  const auto owner = square<TestType>();
  auto failed = tf::cpp::async::isobands(
      owner.mesh(), array<TestType>({0}, {1}), cuts<TestType>());
  CHECK_THROWS_AS(failed.get(), std::invalid_argument);
}

namespace {

template <typename Index, typename Real, std::size_t Ngon>
auto check_python_isobands_fixture() -> void {
  auto owner = python_octahedron<Index, Real, Ngon>();
  const auto mesh = owner.mesh();
  auto scalar_field = array<Real>({-1, 0, 0, 0, 0, 1}, {6});
  auto cut_values = array<Real>({Real{-0.5}, 0, Real{0.5}}, {3});
  auto selected = array<std::int32_t>({1, 2}, {2});

  const auto all = tf::cpp::isobands_with_curves<Index, Real>(
      mesh, scalar_field, cut_values);
  const auto all_without_curves =
      tf::cpp::isobands<Index, Real>(mesh, scalar_field, cut_values);
  const auto python_default_bands = array<std::int32_t>({0, 1, 2, 3}, {4});
  const auto python_default =
      tf::cpp::isobands_with_curves_selected<Index, Real>(
          mesh, scalar_field, cut_values, python_default_bands);
  const auto middle = tf::cpp::isobands_with_curves_selected<Index, Real>(
      mesh, scalar_field, cut_values, selected);
  const auto middle_without_curves = tf::cpp::isobands_selected<Index, Real>(
      mesh, scalar_field, cut_values, selected);

  static_assert(std::is_same_v<
                decltype(all),
                const tf::cpp::isobands_with_curves_result<Index, Real, Ngon>>);
  static_assert(std::is_same_v<decltype(all.mesh),
                               tf::polygons_buffer<Index, Real, 3, Ngon>>);
  static_assert(std::is_same_v<decltype(all.labels), tf::cpp::nd_array<Index>>);
  static_assert(
      std::is_same_v<decltype(all.face_labels), tf::cpp::nd_array<Index>>);
  static_assert(
      std::is_same_v<decltype(all.curves), tf::curves_buffer<Index, Real, 3>>);

  REQUIRE(all.mesh.size() == 24);
  REQUIRE(all.mesh.points_buffer().size() == 18);
  REQUIRE(typed_curves_aligned<Index, Real>(all.curves));
  CHECK(all.curves.size() == 3);
  CHECK(same_array(
      tf::cpp::test::face_indices_of(all.mesh),
      array<Index>({6,  0,  7,  7,  0,  8,  8,  0,  9,  9,  0,  6,  7,  1,  2,
                    7,  2,  6,  8,  3,  1,  8,  1,  7,  9,  4,  3,  9,  3,  8,
                    6,  2,  4,  6,  4,  9,  10, 11, 12, 10, 12, 13, 13, 12, 14,
                    13, 14, 15, 15, 14, 16, 15, 16, 17, 17, 16, 11, 17, 11, 10,
                    13, 5,  10, 15, 5,  13, 17, 5,  15, 10, 5,  17},
                   {72})));
  CHECK(same_array(
      result_points(all.mesh),
      array<Real>(
          {0,          0,  -1,         0,          -1,         0,
           1,          0,  0,          -1,         0,          0,
           0,          1,  0,          0,          0,          1,
           Real{0.5},  0,  Real{-0.5}, 0,          Real{-0.5}, Real{-0.5},
           Real{-0.5}, 0,  Real{-0.5}, 0,          Real{0.5},  Real{-0.5},
           Real{0.5},  0,  Real{0.5},  1,          0,          0,
           0,          -1, 0,          0,          Real{-0.5}, Real{0.5},
           -1,         0,  0,          Real{-0.5}, 0,          Real{0.5},
           0,          1,  0,          0,          Real{0.5},  Real{0.5}},
          {18, 3})));
  CHECK(
      same_array(all.labels, array<Index>({0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3},
                                          {24})));
  CHECK(same_array(all.face_labels,
                   array<Index>({3, 2, 1, 0, 3, 3, 2, 2, 1, 1, 0, 0,
                                 7, 7, 6, 6, 5, 5, 4, 4, 7, 6, 5, 4},
                                {24})));
  CHECK(same_array(tf::cpp::test::arrays_of(all.curves).offsets,
                   array<Index>({0, 5, 10, 15}, {4})));
  CHECK(same_array(
      tf::cpp::test::arrays_of(all.curves).points,
      array<Real>(
          {Real{0.5},  0,  Real{-0.5}, 0,          Real{-0.5}, Real{-0.5},
           Real{-0.5}, 0,  Real{-0.5}, 0,          Real{0.5},  Real{-0.5},
           Real{0.5},  0,  Real{0.5},  1,          0,          0,
           0,          -1, 0,          0,          Real{-0.5}, Real{0.5},
           -1,         0,  0,          Real{-0.5}, 0,          Real{0.5},
           0,          1,  0,          0,          Real{0.5},  Real{0.5}},
          {12, 3})));
  CHECK(same_typed_mesh(all.mesh, all_without_curves.mesh));
  CHECK(same_array(all.labels, all_without_curves.labels));
  CHECK(same_array(all.face_labels, all_without_curves.face_labels));
  CHECK(same_typed_result<Index, Real>(all, python_default));
  const auto duplicate_cut_values =
      array<Real>({Real{-0.5}, 0, 0, Real{0.5}}, {4});
  const auto deduplicated = tf::cpp::isobands_with_curves<Index, Real>(
      mesh, scalar_field, duplicate_cut_values);
  CHECK(same_typed_result<Index, Real>(all, deduplicated));

  REQUIRE(middle.mesh.size() == 16);
  REQUIRE(middle.mesh.points_buffer().size() == 16);
  REQUIRE(typed_curves_aligned<Index, Real>(middle.curves));
  CHECK(middle.curves.size() == 3);
  CHECK(same_array(
      middle.labels,
      array<Index>({1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2}, {16})));
  CHECK(same_array(
      middle.face_labels,
      array<Index>({3, 3, 2, 2, 1, 1, 0, 0, 7, 7, 6, 6, 5, 5, 4, 4}, {16})));
  CHECK(same_array(tf::cpp::test::arrays_of(middle.curves).offsets,
                   array<Index>({0, 5, 10, 15}, {4})));
  CHECK(same_typed_mesh(middle.mesh, middle_without_curves.mesh));
  CHECK(same_array(middle.labels, middle_without_curves.labels));
  CHECK(same_array(middle.face_labels, middle_without_curves.face_labels));

  const auto single_cut = array<Real>({0}, {1});
  const auto two_bands =
      tf::cpp::isobands<Index, Real>(mesh, scalar_field, single_cut);
  std::set<Index> unique_labels(two_bands.labels.begin(),
                                two_bands.labels.end());
  CHECK(unique_labels == std::set<Index>{0, 1});

  // a band is the operand's own, so it is read where the operand was authored
  owner.place({Real{1}, Real{0}, Real{0}, Real{7}, Real{0}, Real{1}, Real{0},
               Real{-4}, Real{0}, Real{0}, Real{1}, Real{2}, Real{0}, Real{0},
               Real{0}, Real{1}});
  const auto raw = tf::cpp::isobands_with_curves<Index, Real>(
      owner.mesh(), scalar_field, cut_values);
  CHECK(same_typed_result<Index, Real>(all, raw));
}

template <typename Index, typename Real, std::size_t Ngon>
auto check_typed_empty_validation_and_async() -> void {
  const auto empty = typed_empty_mesh<Index, Real, Ngon>();
  auto empty_scalars = array<Real>({}, {0});
  auto cut_values = array<Real>({0}, {1});
  auto empty_result = tf::cpp::isobands_with_curves<Index, Real>(
      empty.mesh(), empty_scalars, cut_values);
  CHECK(empty_result.mesh.size() == 0);
  CHECK(empty_result.mesh.points_buffer().size() == 0);
  CHECK(empty_result.labels.length() == 0);
  CHECK(empty_result.face_labels.length() == 0);
  REQUIRE(typed_curves_aligned<Index, Real>(empty_result.curves));
  CHECK(empty_result.curves.size() == 0);
  CHECK(tf::cpp::test::arrays_of(empty_result.curves).ids.empty());

  const auto owner = python_octahedron<Index, Real, Ngon>();
  const auto mesh = owner.mesh();
  auto scalar_field = array<Real>({-1, 0, 0, 0, 0, 1}, {6});
  auto cuts = array<Real>({Real{-0.5}, 0, Real{0.5}}, {3});
  auto selected = array<std::int32_t>({1, 2}, {2});

  CHECK_THROWS_AS((tf::cpp::isobands<Index, Real>(
                      mesh, array<Real>({-1, 0, 0}, {3}), cuts)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::isobands<Index, Real>(mesh, scalar_field,
                                                  array<Real>({0, 1}, {1, 2}))),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::isobands<Index, Real>(
          mesh, scalar_field,
          array<Real>({std::numeric_limits<Real>::infinity()}, {1}))),
      std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::isobands_selected<Index, Real>(
                      mesh, scalar_field, cuts, array<std::int32_t>({4}, {1}))),
                  std::out_of_range);

  const auto baseline = tf::cpp::isobands_with_curves_selected<Index, Real>(
      mesh, scalar_field, cuts, selected);
  auto submissions = std::make_shared<std::atomic<int>>(0);
  auto future = tf::cpp::async::isobands_with_curves_selected(
      mutating_resolver{submissions,
                        [&] {
                          scalar_field.destroy();
                          cuts.destroy();
                          selected.destroy();
                        }},
      mesh, scalar_field, cuts, selected);
  CHECK(same_typed_result<Index, Real>(baseline, future.get()));
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
}

} // namespace

TEST_CASE("isobands matches the exact Python fixed and dynamic dtype matrix",
          "[cpp][iso][isobands][python-fixture][typed]") {
  constexpr auto mixed = tf::dynamic_size;
  check_python_isobands_fixture<std::int32_t, float, 3>();
  check_python_isobands_fixture<std::int64_t, float, 3>();
  check_python_isobands_fixture<std::int32_t, double, 3>();
  check_python_isobands_fixture<std::int64_t, double, 3>();
  check_python_isobands_fixture<std::int32_t, float, mixed>();
  check_python_isobands_fixture<std::int64_t, float, mixed>();
  check_python_isobands_fixture<std::int32_t, double, mixed>();
  check_python_isobands_fixture<std::int64_t, double, mixed>();
}

TEST_CASE(
    "typed isobands preserve empty carriers validation and async snapshots",
    "[cpp][iso][isobands][typed][empty][validation][async]") {
  constexpr auto mixed = tf::dynamic_size;
  check_typed_empty_validation_and_async<std::int32_t, float, 3>();
  check_typed_empty_validation_and_async<std::int64_t, float, 3>();
  check_typed_empty_validation_and_async<std::int32_t, double, 3>();
  check_typed_empty_validation_and_async<std::int64_t, double, 3>();
  check_typed_empty_validation_and_async<std::int32_t, float, mixed>();
  check_typed_empty_validation_and_async<std::int64_t, float, mixed>();
  check_typed_empty_validation_and_async<std::int32_t, double, mixed>();
  check_typed_empty_validation_and_async<std::int64_t, double, mixed>();
}
