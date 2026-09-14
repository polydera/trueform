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

#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/spatial/intersects.hpp"
#include "trueform/cpp/spatial/neighbor_search.hpp"
#include "trueform/cpp/spatial/ray_cast.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace {

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real>
auto primitive(tf::cpp::primitive_kind kind, std::initializer_list<Real> values,
               tf::small_vector<int, 3> shape) -> tf::cpp::primitive<Real> {
  return tf::cpp::primitive<Real>(kind,
                                  make_array<Real>(values, std::move(shape)));
}

template <typename Real>
auto point(Real x, Real y, Real z) -> tf::cpp::primitive<Real> {
  return primitive<Real>(tf::cpp::primitive_kind::point, {x, y, z}, {3});
}

template <typename Real>
auto ray(Real x, Real y, Real z, Real dx, Real dy, Real dz)
    -> tf::cpp::primitive<Real> {
  return primitive<Real>(tf::cpp::primitive_kind::ray, {x, y, z, dx, dy, dz},
                         {2, 3});
}

template <typename Real>
auto square_mesh(Real z = Real{0})
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 2, 3}, {0, 0, z, 1, 0, z, 1, 1, z, 0, 1, z})};
}

template <typename Real>
auto negative_triangle_mesh()
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {0, 0, 0, -1, 0, 0, 0, -1, 0})};
}

template <typename Real>
auto positive_triangle_mesh()
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {2, 0, 0, 3, 0, 0, 2, 1, 0})};
}

template <typename Real>
auto grid_cloud() -> tf::cpp::test::owned_point_cloud<Real> {
  return {tf::cpp::test::points_of<Real>({0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 1,
                                          1, 0, 2, 1, 0, 0, 2, 0, 1, 2, 0, 2, 2,
                                          0})};
}

/// The placement a carrier is assembled at, in the shape `place` states it.
template <typename Real>
auto translation(Real x, Real y, Real z) -> std::array<Real, 16> {
  return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
}

template <typename Real> auto quarter_turn_z() -> std::array<Real, 16> {
  return {0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

template <typename Real> auto identity_placement() -> std::array<Real, 16> {
  return translation<Real>(0, 0, 0);
}

template <typename T>
auto require_shape(const tf::cpp::nd_array<T> &array,
                   std::initializer_list<int> expected) -> void {
  REQUIRE(array.ndim() == static_cast<int>(expected.size()));
  auto dimension = 0;
  for (const auto extent : expected)
    CHECK(array.shape_at(dimension++) == extent);
}

template <typename Real>
auto check_point(const tf::cpp::nd_array<Real> &values, std::size_t offset,
                 Real x, Real y, Real z) -> void {
  CHECK(values[offset] == Catch::Approx(x));
  CHECK(values[offset + 1] == Catch::Approx(y));
  CHECK(values[offset + 2] == Catch::Approx(z));
}

template <typename Real> constexpr auto margin() -> double {
  return std::is_same<Real, float>::value ? 1e-5 : 1e-11;
}

} // namespace

TEMPLATE_TEST_CASE(
    "Python mesh intersection fixtures preserve form transforms and tree reuse",
    "[cpp][spatial][python-parity][form][intersects]", float, double) {
  auto storage0 = square_mesh<TestType>();
  auto storage1 = square_mesh<TestType>();

  CHECK(tf::cpp::intersects(storage0.mesh(), storage1.mesh()).scalar());
  CHECK(tf::cpp::intersects(storage1.mesh(), storage0.mesh()).scalar());
  REQUIRE(storage0.cache.tree_build_count() == 1);
  REQUIRE(storage1.cache.tree_build_count() == 1);

  // a cache is built in local coordinates, so a placement is not a change
  storage0.place(translation<TestType>(10, 5, 2));
  storage1.place(translation<TestType>(10, 5, 2));
  CHECK(tf::cpp::intersects(storage0.mesh(), storage1.mesh()).scalar());
  CHECK(tf::cpp::intersects(storage1.mesh(), storage0.mesh()).scalar());
  CHECK(storage0.cache.tree_build_count() == 1);
  CHECK(storage1.cache.tree_build_count() == 1);

  storage1.place(translation<TestType>(10, 5, 7));
  CHECK_FALSE(tf::cpp::intersects(storage0.mesh(), storage1.mesh()).scalar());
  CHECK_FALSE(tf::cpp::intersects(storage1.mesh(), storage0.mesh()).scalar());
  CHECK(storage0.cache.tree_build_count() == 1);
  CHECK(storage1.cache.tree_build_count() == 1);
}

TEMPLATE_TEST_CASE("Python form primitive intersections retain exact batches "
                   "and world transforms",
                   "[cpp][spatial][python-parity][form][intersects][batch]",
                   float, double) {
  const auto mesh_storage = square_mesh<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto mesh_queries =
      primitive<TestType>(tf::cpp::primitive_kind::point,
                          {TestType{0.75}, TestType{0.25}, 0, TestType{0.75},
                           TestType{0.25}, 1, 0, 0, 0},
                          {3, 3});
  auto mesh_hits = tf::cpp::intersects(mesh, mesh_queries);
  REQUIRE(mesh_hits.is_batch());
  require_shape(mesh_hits.batch(), {3});
  CHECK(mesh_hits.batch()[0] == 1);
  CHECK(mesh_hits.batch()[1] == 0);
  CHECK(mesh_hits.batch()[2] == 1);

  auto cloud_storage = grid_cloud<TestType>();
  const auto cloud_queries = primitive<TestType>(
      tf::cpp::primitive_kind::point,
      {1, 1, 0, TestType{0.5}, TestType{0.5}, 0, 2, 2, 0}, {3, 3});
  const auto cloud_hits =
      tf::cpp::intersects(cloud_storage.point_cloud(), cloud_queries);
  REQUIRE(cloud_hits.is_batch());
  require_shape(cloud_hits.batch(), {3});
  CHECK(cloud_hits.batch()[0] == 1);
  CHECK(cloud_hits.batch()[1] == 0);
  CHECK(cloud_hits.batch()[2] == 1);
  REQUIRE(cloud_storage.cache.tree_build_count() == 1);

  cloud_storage.place(translation<TestType>(10, 5, 2));
  const auto transformed_segment = primitive<TestType>(
      tf::cpp::primitive_kind::segment, {11, 6, 1, 11, 6, 3}, {2, 3});
  CHECK(tf::cpp::intersects(cloud_storage.point_cloud(), transformed_segment)
            .scalar());
  CHECK(cloud_storage.cache.tree_build_count() == 1);

  cloud_storage = {};
  CHECK(cloud_hits.batch()[0] == 1);
  CHECK(cloud_hits.batch()[1] == 0);
  CHECK(cloud_hits.batch()[2] == 1);
}

TEMPLATE_TEST_CASE(
    "Python batch neighbor fixtures preserve every current facade output",
    "[cpp][spatial][python-parity][form][neighbor-search][batch]", float,
    double) {
  const auto cloud_storage = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>({0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1})};
  const auto cloud = cloud_storage.point_cloud();
  const auto cloud_queries = primitive<TestType>(
      tf::cpp::primitive_kind::point,
      {TestType{0.1}, 0, 0, TestType{0.9}, 0, 0, 0, TestType{0.9}, 0}, {3, 3});
  const auto cloud_result =
      tf::cpp::neighbor_search_batch(cloud, cloud_queries);
  require_shape(cloud_result.element_ids, {3});
  require_shape(cloud_result.distances, {3});
  require_shape(cloud_result.points, {3, 3});
  CHECK(cloud_result.element_ids[0] == 0);
  CHECK(cloud_result.element_ids[1] == 1);
  CHECK(cloud_result.element_ids[2] == 2);
  for (std::size_t index = 0; index < 3; ++index)
    CHECK(cloud_result.distances[index] ==
          Catch::Approx(TestType{0.01}).margin(margin<TestType>()));
  check_point(cloud_result.points, 0, TestType{0}, TestType{0}, TestType{0});
  check_point(cloud_result.points, 3, TestType{1}, TestType{0}, TestType{0});
  check_point(cloud_result.points, 6, TestType{0}, TestType{1}, TestType{0});
  CHECK(cloud_storage.cache.tree_build_count() == 1);

  const auto mesh_storage = square_mesh<TestType>();
  const auto mesh = mesh_storage.mesh();
  const auto mesh_queries = primitive<TestType>(
      tf::cpp::primitive_kind::point,
      {TestType{0.75}, TestType{0.25}, 1, TestType{0.25}, TestType{0.75}, 2},
      {2, 3});
  const auto mesh_result = tf::cpp::neighbor_search_batch(mesh, mesh_queries);
  require_shape(mesh_result.element_ids, {2});
  require_shape(mesh_result.distances, {2});
  require_shape(mesh_result.points, {2, 3});
  CHECK(mesh_result.element_ids[0] == 0);
  CHECK(mesh_result.element_ids[1] == 1);
  CHECK(mesh_result.distances[0] == Catch::Approx(1));
  CHECK(mesh_result.distances[1] == Catch::Approx(4));
  check_point(mesh_result.points, 0, TestType{0.75}, TestType{0.25},
              TestType{0});
  check_point(mesh_result.points, 3, TestType{0.25}, TestType{0.75},
              TestType{0});
}

TEMPLATE_TEST_CASE(
    "Python transformed neighbor fixtures return world points without "
    "rebuilding",
    "[cpp][spatial][python-parity][form][neighbor-search][transform]", float,
    double) {
  auto storage = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>({1, 0, 0, 2, 0, 0})};
  storage.place(quarter_turn_z<TestType>());

  auto transformed_query = point<TestType>(0, TestType{1.1}, 0);
  const auto transformed =
      tf::cpp::neighbor_search(storage.point_cloud(), transformed_query);
  CHECK(transformed.element_id == 0);
  CHECK(transformed.distance2 ==
        Catch::Approx(TestType{0.01}).margin(margin<TestType>()));
  require_shape(transformed.point, {3});
  check_point(transformed.point, 0, TestType{0}, TestType{1}, TestType{0});
  REQUIRE(storage.cache.tree_build_count() == 1);

  storage.place(identity_placement<TestType>());
  auto original_query = point<TestType>(TestType{1.1}, 0, 0);
  const auto original =
      tf::cpp::neighbor_search(storage.point_cloud(), original_query);
  CHECK(original.element_id == 0);
  CHECK(original.distance2 ==
        Catch::Approx(TestType{0.01}).margin(margin<TestType>()));
  check_point(original.point, 0, TestType{1}, TestType{0}, TestType{0});
  CHECK(storage.cache.tree_build_count() == 1);
}

TEMPLATE_TEST_CASE("Python nearest form pairs preserve IDs squared metrics "
                   "points and symmetry",
                   "[cpp][spatial][python-parity][form][neighbor-search][pair]",
                   float, double) {
  const auto storage0 = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>({0, 0, 0, 4, 0, 0})};
  const auto storage1 = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>(
          {TestType{0.2}, TestType{0.1}, 0, 10, 10, 10})};
  const auto cloud0 = storage0.point_cloud();
  const auto cloud1 = storage1.point_cloud();
  const auto cloud_pair = tf::cpp::neighbor_search(cloud0, cloud1);
  CHECK(cloud_pair.element_id0 == 0);
  CHECK(cloud_pair.element_id1 == 0);
  CHECK(cloud_pair.distance2 ==
        Catch::Approx(TestType{0.05}).margin(margin<TestType>()));
  check_point(cloud_pair.point0, 0, TestType{0}, TestType{0}, TestType{0});
  check_point(cloud_pair.point1, 0, TestType{0.2}, TestType{0.1}, TestType{0});

  const auto reverse_cloud_pair = tf::cpp::neighbor_search(cloud1, cloud0);
  CHECK(reverse_cloud_pair.element_id0 == cloud_pair.element_id1);
  CHECK(reverse_cloud_pair.element_id1 == cloud_pair.element_id0);
  CHECK(reverse_cloud_pair.distance2 ==
        Catch::Approx(cloud_pair.distance2).margin(margin<TestType>()));
  check_point(reverse_cloud_pair.point0, 0, TestType{0.2}, TestType{0.1},
              TestType{0});
  check_point(reverse_cloud_pair.point1, 0, TestType{0}, TestType{0},
              TestType{0});

  const auto mesh_storage = square_mesh<TestType>();
  const auto above_storage = tf::cpp::test::owned_point_cloud<TestType>{
      tf::cpp::test::points_of<TestType>(
          {TestType{0.5}, TestType{0.25}, 3, 10, 10, 10})};
  const auto cloud_mesh = tf::cpp::neighbor_search(
      above_storage.point_cloud(), mesh_storage.mesh());
  CHECK(cloud_mesh.element_id0 == 0);
  CHECK(cloud_mesh.element_id1 == 0);
  CHECK(cloud_mesh.distance2 == Catch::Approx(9));
  check_point(cloud_mesh.point0, 0, TestType{0.5}, TestType{0.25}, TestType{3});
  check_point(cloud_mesh.point1, 0, TestType{0.5}, TestType{0.25}, TestType{0});

  const auto storage_negative = negative_triangle_mesh<TestType>();
  const auto storage_positive = positive_triangle_mesh<TestType>();
  const auto mesh_pair = tf::cpp::neighbor_search(storage_negative.mesh(),
                                                  storage_positive.mesh());
  CHECK(mesh_pair.element_id0 == 0);
  CHECK(mesh_pair.element_id1 == 0);
  CHECK(mesh_pair.distance2 == Catch::Approx(4));
  check_point(mesh_pair.point0, 0, TestType{0}, TestType{0}, TestType{0});
  check_point(mesh_pair.point1, 0, TestType{2}, TestType{0}, TestType{0});
}

TEMPLATE_TEST_CASE("Python form ray batches preserve exact IDs parameters "
                   "misses and ownership",
                   "[cpp][spatial][python-parity][form][ray-cast][batch]",
                   float, double) {
  auto mesh_storage = square_mesh<TestType>();
  auto mesh_rays = primitive<TestType>(tf::cpp::primitive_kind::ray,
                                       {TestType{0.75}, TestType{0.25}, 2, 0, 0,
                                        -1, 5, 5, 2, 0, 0, -1, TestType{0.25},
                                        TestType{0.75}, 1, 0, 0, -1},
                                       {3, 2, 3});
  auto mesh_result =
      tf::cpp::ray_cast(mesh_rays, mesh_storage.mesh()).batch();
  require_shape(mesh_result.hits, {3});
  require_shape(mesh_result.ts, {3});
  require_shape(mesh_result.element_ids, {3});
  CHECK(mesh_result.hits[0] == 1);
  CHECK(mesh_result.hits[1] == 0);
  CHECK(mesh_result.hits[2] == 1);
  CHECK(mesh_result.element_ids[0] == 0);
  CHECK(mesh_result.element_ids[1] == -1);
  CHECK(mesh_result.element_ids[2] == 1);
  CHECK(mesh_result.ts[0] == Catch::Approx(2));
  CHECK(mesh_result.ts[1] == TestType{0});
  CHECK(mesh_result.ts[2] == Catch::Approx(1));

  auto cloud_storage = grid_cloud<TestType>();
  auto cloud_rays =
      primitive<TestType>(tf::cpp::primitive_kind::ray,
                          {1, 1, 2, 0, 0, -1, TestType{0.5}, TestType{0.5}, 2,
                           0, 0, -1, 2, 0, 3, 0, 0, -1},
                          {3, 2, 3});
  auto cloud_result =
      tf::cpp::ray_cast(cloud_rays, cloud_storage.point_cloud()).batch();
  require_shape(cloud_result.hits, {3});
  require_shape(cloud_result.ts, {3});
  require_shape(cloud_result.element_ids, {3});
  CHECK(cloud_result.hits[0] == 1);
  CHECK(cloud_result.hits[1] == 0);
  CHECK(cloud_result.hits[2] == 1);
  CHECK(cloud_result.element_ids[0] == 4);
  CHECK(cloud_result.element_ids[1] == -1);
  CHECK(cloud_result.element_ids[2] == 2);
  CHECK(cloud_result.ts[0] == Catch::Approx(2));
  CHECK(cloud_result.ts[1] == TestType{0});
  CHECK(cloud_result.ts[2] == Catch::Approx(3));

  mesh_storage = {};
  cloud_storage = {};
  mesh_rays.data().destroy();
  cloud_rays.data().destroy();
  CHECK(mesh_result.element_ids[2] == 1);
  CHECK(mesh_result.ts[0] == Catch::Approx(2));
  CHECK(cloud_result.element_ids[0] == 4);
  CHECK(cloud_result.ts[2] == Catch::Approx(3));
}

TEMPLATE_TEST_CASE(
    "Empty Python forms produce scalar intersection and ray misses safely",
    "[cpp][spatial][python-parity][form][empty]", float, double) {
  const auto mesh_storage =
      tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType>{};
  const auto cloud_storage = tf::cpp::test::owned_point_cloud<TestType>{};
  const auto mesh = mesh_storage.mesh();
  const auto cloud = cloud_storage.point_cloud();
  const auto query = point<TestType>(0, 0, 0);
  CHECK_FALSE(tf::cpp::intersects(mesh, query).scalar());
  CHECK_FALSE(tf::cpp::intersects(cloud, query).scalar());

  const auto query_ray = ray<TestType>(0, 0, -1, 0, 0, 1);
  const auto mesh_miss = tf::cpp::ray_cast(query_ray, mesh).scalar();
  const auto cloud_miss = tf::cpp::ray_cast(query_ray, cloud).scalar();
  CHECK_FALSE(mesh_miss.hit);
  CHECK(mesh_miss.element_id == -1);
  CHECK(mesh_miss.t == TestType{0});
  CHECK_FALSE(cloud_miss.hit);
  CHECK(cloud_miss.element_id == -1);
  CHECK(cloud_miss.t == TestType{0});
}
