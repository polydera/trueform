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

#include "trueform/core/unit_vectors_buffer.hpp"
#include "trueform/cpp/core/build_tree.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/core/point_cloud_cache.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace {

template <typename Real, std::size_t Dims>
auto cloud_translation() -> std::array<Real, (Dims + 1) * (Dims + 1)> {
  constexpr auto side = Dims + 1;
  std::array<Real, side * side> values{};
  for (std::size_t row = 0; row != side; ++row)
    values[row * side + row] = Real{1};
  values[Dims] = Real{5};
  return values;
}

template <typename Real, std::size_t Dims>
auto check_point_cloud_carrier_matrix_combination() -> void {
  INFO("Real bytes = " << sizeof(Real) << ", Dims = " << Dims);
  using owned_t = tf::cpp::test::owned_point_cloud<Real, Dims>;

  auto owned = [] {
    if constexpr (Dims == 2)
      return owned_t{tf::cpp::test::points_of<Real, 2>({0, 0, 2, 3})};
    else
      return owned_t{tf::cpp::test::points_of<Real, 3>({0, 0, 0, 2, 3, 4})};
  }();

  CHECK(owned.point_cloud().number_of_points() == 2);
  CHECK(owned.point_cloud().frame()(0, static_cast<int>(Dims)) == Real{0});

  owned.place(cloud_translation<Real, Dims>());
  const auto cloud = owned.point_cloud();
  CHECK(cloud.frame()(0, static_cast<int>(Dims)) == Real{5});

  tf::cpp::build_tree(cloud);
  REQUIRE(owned.cache.is_tree_fresh(cloud.geometry()));
  CHECK(cloud.tree().bv().max[Dims - 1] == (Dims == 2 ? Real{3} : Real{4}));

  // A CACHE IS A VALUE: a copy shares nothing, so it has no tree until one is
  // asked of it, and it built none of what it carries
  const auto copy = owned.cache;
  CHECK_FALSE(copy.is_tree_built());
  CHECK(copy.tree_build_count() == 0);
  // a cloud's link has no builder, so a cloud that was never handed one says
  // so with the word every carrier says it with
  CHECK_FALSE(copy.is_vertex_link_fresh(cloud.geometry()));
}

using default_cloud = tf::cpp::point_cloud<float>;
using explicit_dims_cloud = tf::cpp::point_cloud<float, 3>;
using cloud_2d = tf::cpp::point_cloud<double, 2>;
using cloud_3d = tf::cpp::point_cloud<double, 3>;

static_assert(std::is_same_v<default_cloud, explicit_dims_cloud>);
static_assert(std::is_same_v<tf::cpp::point_cloud_cache<float>,
                             tf::cpp::point_cloud_cache<float, 3>>);
static_assert(std::is_same_v<typename default_cloud::cache_type,
                             tf::cpp::point_cloud_cache<float, 3>>);
static_assert(!std::is_default_constructible_v<cloud_2d>);
static_assert(std::is_copy_constructible_v<cloud_2d>);
static_assert(
    !std::is_same_v<decltype(std::declval<const cloud_2d &>().points()),
                    decltype(std::declval<const cloud_3d &>().points())>);
static_assert(
    !std::is_same_v<decltype(std::declval<const cloud_2d &>().normals()),
                    decltype(std::declval<const cloud_3d &>().normals())>);
static_assert(
    !std::is_same_v<decltype(std::declval<const cloud_2d &>().tree()),
                    decltype(std::declval<const cloud_3d &>().tree())>);

} // namespace

TEST_CASE("approved point-cloud carrier matrix links with dimensional state",
          "[cpp][core][point-cloud][templates][matrix]") {
  check_point_cloud_carrier_matrix_combination<float, 2>();
  check_point_cloud_carrier_matrix_combination<float, 3>();
  check_point_cloud_carrier_matrix_combination<double, 2>();
  check_point_cloud_carrier_matrix_combination<double, 3>();
}

TEST_CASE("a cloud's normals ride the reading its points state",
          "[cpp][core][point-cloud][templates]") {
  tf::cpp::test::owned_point_cloud<float> owned{
      tf::cpp::test::points_of<float>({0, 0, 0, 2, 0, 0, 0, 3, 0})};
  owned.normals.allocate(3);
  auto &normal_storage = owned.normals.data_buffer();
  for (std::size_t point = 0; point != 3; ++point) {
    normal_storage[point * 3] = 0;
    normal_storage[point * 3 + 1] = 0;
    normal_storage[point * 3 + 2] = 1;
  }

  CHECK(owned.point_cloud().number_of_points() == 3);
  CHECK(owned.point_cloud().has_normals());
  CHECK_FALSE(owned.cache.is_tree_built());
  CHECK(owned.cache.tree_build_count() == 0);

  tf::cpp::build_tree(owned.point_cloud());
  REQUIRE(owned.cache.is_tree_fresh(owned.point_cloud().geometry()));
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(owned.point_cloud().tree().bv().min[0] == 0);
  CHECK(owned.point_cloud().tree().bv().max[0] == 2);
  tf::cpp::build_tree(owned.point_cloud());
  CHECK(owned.cache.tree_build_count() == 1);

  owned.place({1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});

  // the frame is always tagged, so a form has one type whatever placed it, and
  // a cloud assembled without a frame reads as identity
  const auto placed = owned.point_cloud();
  const auto form = placed.form();
  static_assert(tf::has_frame_policy<decltype(form)>);
  CHECK(form[1][0] == 2);
  CHECK(form.tree().bv().max[0] == 2);
  CHECK(form.transformation()(0, 3) == 5);

  // the caller moves its own points and states it; the next reading is not the
  // one the tree was built for
  auto &points = owned.points.data_buffer();
  points[0] = 10;
  points[3] = 12;
  points[6] = 10;
  owned.cache.points_changed();
  const auto moved = owned.point_cloud();
  CHECK_FALSE(owned.cache.is_tree_fresh(moved.geometry()));
  CHECK(moved.has_normals());
  tf::cpp::build_tree(moved);
  CHECK(owned.cache.is_tree_fresh(moved.geometry()));
  CHECK(owned.cache.tree_build_count() == 2);
  CHECK(moved.tree().bv().min[0] == 10);
}
