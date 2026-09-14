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
#include "trueform/core/buffer.hpp"
#include "trueform/core/points_buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/cache.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <tuple>

namespace {

template <typename Index, typename Real, std::size_t Dims> struct door_row {
  using real_type = Real;
  using index_type = Index;
  static constexpr auto dims = Dims;
};

using door_rows = std::tuple<
    door_row<std::int32_t, float, 2>, door_row<std::int32_t, float, 3>,
    door_row<std::int64_t, float, 2>, door_row<std::int64_t, float, 3>,
    door_row<std::int32_t, double, 2>, door_row<std::int32_t, double, 3>,
    door_row<std::int64_t, double, 2>, door_row<std::int64_t, double, 3>>;

template <typename T>
auto door_assign(tf::buffer<T> &buffer, std::initializer_list<T> values)
    -> T * {
  buffer.allocate(values.size());
  std::copy(values.begin(), values.end(), buffer.begin());
  return buffer.begin();
}

template <typename Real, std::size_t Dims>
auto door_assign_points(tf::points_buffer<Real, Dims> &points,
                        std::size_t point_count) -> Real * {
  points.allocate(point_count);
  auto &data = points.data_buffer();
  for (std::size_t point = 0; point < point_count; ++point)
    for (std::size_t dim = 0; dim < Dims; ++dim)
      data[point * Dims + dim] = static_cast<Real>(point * (Dims + 1) + dim);
  return data.begin();
}

template <typename Index, typename Real, std::size_t Dims>
auto door_mixed_polygons(std::initializer_list<Index> offsets,
                         std::initializer_list<Index> indices,
                         std::size_t point_count)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> polygons;
  door_assign(polygons.faces_buffer().offsets_buffer(), offsets);
  door_assign(polygons.faces_buffer().data_buffer(), indices);
  door_assign_points(polygons.points_buffer(), point_count);
  return polygons;
}

/// The door is the CACHE's, asked once per reading — so a test states a new
/// reading of the storage it wants refused, and asks.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto door_reads(tf::cpp::cache<Index, Real, Dims, Ngon> &cache,
                const tf::polygons_buffer<Index, Real, Dims, Ngon> &polygons)
    -> void {
  cache.faces_changed();
  cache.points_changed();
  tf::cpp::mesh<Index, Real, Dims, Ngon>(polygons.faces(), polygons.points(),
                                         cache)
      .require_indices();
}

} // namespace

TEMPLATE_LIST_TEST_CASE("a mesh reads core's polygons where they lie",
                        "[cpp][core][mesh][door][matrix]", door_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  const auto polygons = door_mixed_polygons<Index, Real, Dims>(
      {Index{0}, Index{3}, Index{7}},
      {Index{0}, Index{1}, Index{2}, Index{1}, Index{4}, Index{3}, Index{2}},
      5);
  tf::cpp::cache<Index, Real, Dims, tf::dynamic_size> cache;
  const auto mesh = tf::cpp::mesh<Index, Real, Dims, tf::dynamic_size>(
      polygons.faces(), polygons.points(), cache);

  // nothing was copied and nothing moved: the caller keeps its storage and the
  // mesh is one reading of it
  CHECK(mesh.number_of_faces() == 2);
  CHECK(mesh.number_of_points() == 5);
  CHECK(&mesh.faces()[0][0] == polygons.faces_buffer().data_buffer().data());
  CHECK(&mesh.points()[0][0] == polygons.points_buffer().data_buffer().data());

  const auto faces = mesh.faces();
  REQUIRE(faces.size() == 2);
  CHECK(faces[0].size() == 3);
  CHECK(faces[1].size() == 4);
  CHECK(faces[1][1] == Index{4});

  const auto membership = mesh.face_membership();
  REQUIRE(membership.size() == 5);
  REQUIRE(membership[0].size() == 1);
  CHECK(membership[0][0] == Index{0});
  CHECK(membership[4][0] == Index{1});
  CHECK(cache.is_face_membership_fresh(mesh.geometry()));
  CHECK(cache.face_membership_build_count() == 1);
}

TEMPLATE_LIST_TEST_CASE("a triangle mesh states three consecutive sides",
                        "[cpp][core][mesh][door][matrix]", door_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  tf::polygons_buffer<Index, Real, Dims, 3> polygons;
  polygons.faces_buffer().allocate(1);
  door_assign(polygons.faces_buffer().data_buffer(),
              {Index{0}, Index{1}, Index{2}});
  door_assign_points(polygons.points_buffer(), 3);

  const auto &stored = polygons;
  tf::cpp::cache<Index, Real, Dims, 3> cache;
  const auto mesh = tf::cpp::mesh<Index, Real, Dims, 3>(
      stored.faces(), stored.points(), cache);
  CHECK(&mesh.faces()[0][0] == stored.faces_buffer().data_buffer().data());
  CHECK(&mesh.points()[0][0] == stored.points_buffer().data_buffer().data());
  CHECK(mesh.number_of_faces() == 1);
}

TEMPLATE_LIST_TEST_CASE("an empty mixed mesh states no offsets at all",
                        "[cpp][core][mesh][door][empty][matrix]", door_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  const auto canonical =
      door_mixed_polygons<Index, Real, Dims>({Index{0}}, {}, std::size_t{0});
  tf::cpp::cache<Index, Real, Dims, tf::dynamic_size> first;
  const auto stated = tf::cpp::mesh<Index, Real, Dims, tf::dynamic_size>(
      canonical.faces(), canonical.points(), first);
  CHECK(stated.number_of_faces() == 0);
  CHECK(stated.number_of_points() == 0);

  // a caller that states no offsets at all states the same empty mesh
  const tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size> nothing;
  tf::cpp::cache<Index, Real, Dims, tf::dynamic_size> second;
  const auto empty = tf::cpp::mesh<Index, Real, Dims, tf::dynamic_size>(
      nothing.faces(), nothing.points(), second);
  CHECK(empty.number_of_faces() == 0);
  CHECK(empty.number_of_points() == 0);
  CHECK(empty.faces().size() == 0);
  CHECK_NOTHROW(empty.require_indices());
}

TEMPLATE_LIST_TEST_CASE("the door refuses what raw storage states wrongly",
                        "[cpp][core][mesh][door][validation][matrix]",
                        door_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  tf::cpp::cache<Index, Real, Dims, tf::dynamic_size> cache;
  const auto wrong_start = door_mixed_polygons<Index, Real, Dims>(
      {Index{1}, Index{4}}, {Index{0}, Index{1}, Index{2}, Index{0}}, 3);
  CHECK_THROWS_AS(door_reads(cache, wrong_start),
                  std::invalid_argument);

  const auto wrong_end = door_mixed_polygons<Index, Real, Dims>(
      {Index{0}, Index{3}}, {Index{0}, Index{1}, Index{2}, Index{0}}, 3);
  CHECK_THROWS_AS(door_reads(cache, wrong_end),
                  std::invalid_argument);

  const auto decreasing = door_mixed_polygons<Index, Real, Dims>(
      {Index{0}, Index{4}, Index{3}}, {Index{0}, Index{1}, Index{2}}, 3);
  CHECK_THROWS_AS(door_reads(cache, decreasing),
                  std::invalid_argument);

  const auto short_face = door_mixed_polygons<Index, Real, Dims>(
      {Index{0}, Index{2}}, {Index{0}, Index{1}}, 2);
  CHECK_THROWS_AS(door_reads(cache, short_face),
                  std::invalid_argument);

  const auto negative = door_mixed_polygons<Index, Real, Dims>(
      {Index{0}, Index{3}}, {Index{0}, Index{-1}, Index{2}}, 3);
  CHECK_THROWS_AS(door_reads(cache, negative), std::out_of_range);

  const auto upper_bound = door_mixed_polygons<Index, Real, Dims>(
      {Index{0}, Index{3}}, {Index{0}, Index{1}, Index{3}}, 3);
  CHECK_THROWS_AS(door_reads(cache, upper_bound),
                  std::out_of_range);

  // a refusal leaves the caller's storage exactly where it was, because the
  // door never had it
  CHECK(upper_bound.faces_buffer().offsets_buffer().size() == 2);
  CHECK(upper_bound.faces_buffer().data_buffer().size() == 3);
  CHECK(upper_bound.points_buffer().data_buffer().size() == 3 * Dims);

  auto misaligned_points = door_mixed_polygons<Index, Real, Dims>(
      {Index{0}, Index{3}}, {Index{0}, Index{1}, Index{2}}, 3);
  misaligned_points.points_buffer().data_buffer().reallocate(3 * Dims + 1);
  CHECK_THROWS_AS(door_reads(cache, misaligned_points),
                  std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE("the door refuses a triangle mesh's own faces too",
                        "[cpp][core][mesh][door][validation][matrix]",
                        door_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  tf::cpp::cache<Index, Real, Dims, 3> cache;
  tf::polygons_buffer<Index, Real, Dims, 3> negative;
  negative.faces_buffer().allocate(1);
  door_assign(negative.faces_buffer().data_buffer(),
              {Index{0}, Index{-1}, Index{2}});
  door_assign_points(negative.points_buffer(), 3);
  CHECK_THROWS_AS(door_reads(cache, negative), std::out_of_range);

  tf::polygons_buffer<Index, Real, Dims, 3> upper_bound;
  upper_bound.faces_buffer().allocate(1);
  door_assign(upper_bound.faces_buffer().data_buffer(),
              {Index{0}, Index{1}, Index{3}});
  door_assign_points(upper_bound.points_buffer(), 3);
  CHECK_THROWS_AS(door_reads(cache, upper_bound),
                  std::out_of_range);
  CHECK(upper_bound.faces_buffer().data_buffer().size() == 3);
}
