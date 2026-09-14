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

#include "trueform/cpp/core/index_map.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/core/reductions.hpp"

#include <catch2/catch_test_macros.hpp>

#include <tbb/global_control.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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

template <typename Index>
auto make_blocks(std::initializer_list<Index> offsets,
                 std::initializer_list<Index> data)
    -> tf::cpp::offset_blocked_buffer<Index, Index> {
  return tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>(offsets, {static_cast<int>(offsets.size())}),
      make_array<Index>(data, {static_cast<int>(data.size())}));
}

template <typename T> auto check_deep_copy_view() -> void {
  auto array = make_array<T>({T{0}, T{1}, T{2}, T{3}, T{4}, T{5}, T{6}, T{7},
                              T{8}, T{9}, T{10}, T{11}},
                             {4, 3});
  auto view = array.slice(1, 3);
  auto copy = view.deep_copy();

  REQUIRE((copy.raw_shape() == tf::small_vector<int, 3>{2, 3}));
  REQUIRE(copy.length() == 6);
  CHECK(view.raw_data() == array.raw_data() + 3);
  CHECK(copy.raw_data() != view.raw_data());
  for (std::size_t index = 0; index < copy.length(); ++index)
    CHECK(copy[index] == static_cast<T>(index + 3));

  copy[0] = T{42};
  CHECK(view[0] == T{3});
  view[1] = T{43};
  CHECK(copy[1] == T{4});

  auto empty = make_array<T>({}, {0});
  auto empty_copy = empty.deep_copy();
  CHECK(empty_copy.is_valid());
  CHECK(empty_copy.empty());
  CHECK((empty_copy.raw_shape() == tf::small_vector<int, 3>{0}));
}

template <typename T> auto deep_copy_test_value(std::size_t index) -> T {
  return static_cast<T>(static_cast<int>(index % 97U) - 48);
}

template <typename T>
auto check_deep_copy_crossover_size(std::size_t length) -> void {
  tf::buffer<T> buffer;
  buffer.allocate(length + 2);
  for (std::size_t index = 0; index < buffer.size(); ++index)
    buffer[index] = deep_copy_test_value<T>(index);

  auto array = tf::cpp::nd_array<T>::from_buffer(
      std::move(buffer), {static_cast<int>(length + 2)});
  auto view = array.slice(1, static_cast<int>(length + 1));
  REQUIRE(view.raw_data() == array.raw_data() + 1);
  REQUIRE(view.length() == length);

  auto copy = view.deep_copy();
  REQUIRE(copy.is_valid());
  REQUIRE(copy.length() == length);
  REQUIRE(
      (copy.raw_shape() == tf::small_vector<int, 3>{static_cast<int>(length)}));
  REQUIRE(copy.raw_data() != view.raw_data());
  for (std::size_t index = 0; index < length; ++index)
    REQUIRE(copy[index] == deep_copy_test_value<T>(index + 1));
}

template <typename T> auto check_deep_copy_crossover(int thread_cap) -> void {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                              static_cast<std::size_t>(thread_cap));
  REQUIRE(tbb::global_control::active_value(
              tbb::global_control::max_allowed_parallelism) ==
          static_cast<std::size_t>(thread_cap));

  constexpr auto serial_copy_max_bytes = std::size_t{1} * 1024U * 1024U;
  constexpr auto serial_max_elements = serial_copy_max_bytes / sizeof(T);
  check_deep_copy_crossover_size<T>(serial_max_elements);
  check_deep_copy_crossover_size<T>(serial_max_elements + 1);
}

template <typename T> auto check_invalid_deep_copy() -> void {
  const tf::cpp::nd_array<T> invalid;
  const auto copy = invalid.deep_copy();
  CHECK_FALSE(copy.is_valid());
  CHECK(copy.raw_data() == nullptr);
  CHECK(copy.length() == 0);
}

auto handles_triangle()
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
      {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
}

static_assert(std::is_same<tf::cpp::nd_array<std::int8_t>,
                           tf::cpp::nd_array<std::int8_t>>::value,
              "boolean arrays retain int8 storage");

} // namespace

TEST_CASE("nd_array views share storage and own shape metadata",
          "[cpp][core][storage]") {
  auto array = make_array<float>({1, 2, 3, 4, 5, 6}, {2, 3});
  auto shallow = array.shallow_copy();

  shallow[0] = 42;
  CHECK(array[0] == 42);
  shallow.set_shape({6});
  CHECK(shallow.ndim() == 1);
  CHECK(array.ndim() == 2);
  CHECK(array.shape_at(0) == 2);

  auto row = array.row(1);
  auto slice = array.slice(0, 1);
  array.destroy();

  CHECK_FALSE(array.is_valid());
  REQUIRE(row.is_valid());
  CHECK(row.length() == 3);
  CHECK(row[0] == 4);
  CHECK(slice.shape_at(0) == 1);
  CHECK(slice[2] == 3);
}

TEST_CASE("nd_array validates shape and view bounds", "[cpp][core][storage]") {
  CHECK_THROWS_AS(make_array<std::int32_t>({1, 2, 3}, {2, 2}),
                  std::invalid_argument);
  CHECK_THROWS_AS(make_array<std::int32_t>({1, 2, 3}, {-1, 3}),
                  std::invalid_argument);

  auto array = make_array<std::int32_t>({1, 2, 3, 4}, {2, 2});
  CHECK_THROWS_AS(array.set_shape({3}), std::invalid_argument);
  CHECK_THROWS_AS(array.row(2), std::out_of_range);
  CHECK_THROWS_AS(array.slice(-1, 1), std::out_of_range);
  CHECK_THROWS_AS(array.slice(1, 3), std::out_of_range);
  CHECK_THROWS_AS(array.shape_at(2), std::out_of_range);
}

TEST_CASE("nd_array deep copies preserve typed views and detach storage",
          "[cpp][core][storage]") {
  check_deep_copy_view<std::int8_t>();
  check_deep_copy_view<std::int32_t>();
  check_deep_copy_view<float>();
  check_deep_copy_view<double>();
}

TEST_CASE("nd_array deep copy crossover preserves nonzero-offset typed views",
          "[cpp][core][storage]") {
  for (const auto thread_cap : {1, 8}) {
    check_deep_copy_crossover<std::int8_t>(thread_cap);
    check_deep_copy_crossover<std::int32_t>(thread_cap);
    check_deep_copy_crossover<float>(thread_cap);
    check_deep_copy_crossover<double>(thread_cap);
  }
}

TEST_CASE("nd_array deep copying an invalid default stays invalid",
          "[cpp][core][storage]") {
  check_invalid_deep_copy<std::int8_t>();
  check_invalid_deep_copy<std::int32_t>();
  check_invalid_deep_copy<float>();
  check_invalid_deep_copy<double>();
}

TEST_CASE("nd_array borrows external memory without copying it",
          "[cpp][core][storage][borrowed]") {
  auto owner = std::make_shared<std::vector<float>>(
      std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  auto *const external = owner->data();
  auto array = tf::cpp::nd_array<float>::from_borrowed(owner, external,
                                                       owner->size(), {4, 3});

  REQUIRE(array.is_valid());
  CHECK(array.raw_data() == external);
  CHECK(array.length() == 12);
  CHECK(array.shape_at(1) == 3);
  CHECK(array[5] == 5.0F);

  array[0] = 42.0F;
  CHECK((*owner)[0] == 42.0F);

  auto view = array.slice(1, 3);
  CHECK(view.raw_data() == external + 3);
  CHECK(view.length() == 6);

  auto row = array.row(2);
  CHECK(row.raw_data() == external + 6);

  const auto copy = array.deep_copy();
  CHECK(copy.raw_data() != external);
  CHECK(copy[0] == 42.0F);
  (*owner)[1] = 99.0F;
  CHECK(copy[1] == 1.0F);
}

TEST_CASE("a borrowed nd_array retains the keepalive it was handed",
          "[cpp][core][storage][borrowed]") {
  auto owner = std::make_shared<std::vector<double>>(64, 7.0);
  REQUIRE(owner.use_count() == 1);

  tf::cpp::nd_array<double> array;
  {
    auto handed = tf::cpp::nd_array<double>::from_borrowed(owner, owner->data(),
                                                           owner->size(), {64});
    CHECK(owner.use_count() == 2);
    array = handed.slice(0, 32);
    CHECK(owner.use_count() == 3);
  }
  CHECK(owner.use_count() == 2);

  const auto *const external = owner->data();
  owner.reset();
  REQUIRE(array.is_valid());
  CHECK(array.raw_data() == external);
  for (std::size_t index = 0; index < array.length(); ++index)
    CHECK(array[index] == 7.0);

  array.destroy();
  CHECK_FALSE(array.is_valid());
}

TEST_CASE("a borrowed nd_array with no keepalive enters the operations",
          "[cpp][core][storage][borrowed]") {
  std::vector<std::int32_t> values{1, 2, 3, 4, 5, 6};
  auto array = tf::cpp::nd_array<std::int32_t>::from_borrowed(
      {}, values.data(), values.size(), {2, 3});

  CHECK(array.raw_data() == values.data());
  CHECK(tf::cpp::sum(array) == 21);

  const auto rows = tf::cpp::sum(array, 1);
  REQUIRE(rows.length() == 2);
  CHECK(rows[0] == 6);
  CHECK(rows[1] == 15);
}

TEST_CASE("a borrowed nd_array validates its shape against its length",
          "[cpp][core][storage][borrowed]") {
  std::vector<float> values{0, 1, 2, 3, 4, 5};
  CHECK_THROWS_AS(tf::cpp::nd_array<float>::from_borrowed(
                      {}, values.data(), values.size(), {2, 2}),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::nd_array<float>::from_borrowed({}, values.data(),
                                                          values.size(), {}),
                  std::invalid_argument);
}

TEST_CASE("offset blocked buffers expose lifetime-safe blocks",
          "[cpp][core][storage]") {
  auto empty =
      tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
          make_array<std::int32_t>({}, {0}), make_array<std::int32_t>({}, {0}));
  CHECK(empty.is_valid());
  CHECK(empty.size() == 0);

  auto offsets = make_array<std::int32_t>({0, 2, 2, 5}, {4});
  auto values = make_array<std::int32_t>({4, 5, 7, 8, 9}, {5});
  auto blocks =
      tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
          std::move(offsets), std::move(values));

  CHECK(blocks.size() == 3);
  CHECK(blocks.get(0).length() == 2);
  CHECK(blocks.get(1).empty());
  auto final_block = blocks.get(2);
  blocks.destroy();
  CHECK(final_block[0] == 7);
  CHECK(final_block[2] == 9);

  auto invalid_offsets = make_array<std::int32_t>({1, 2}, {2});
  auto invalid_values = make_array<std::int32_t>({7, 8}, {2});
  CHECK_THROWS_AS(
      (tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
          std::move(invalid_offsets), std::move(invalid_values))),
      std::invalid_argument);
}

// A copy IS the shallow copy -- the members are shared-storage arrays -- and
// `= {}` is what emptying one looks like.
TEST_CASE("an index map shares storage by value and deep-copies by ask",
          "[cpp][core][storage]") {
  tf::cpp::index_map<> map{make_array<std::int32_t>({0, -1, 1}, {3}),
                           make_array<std::int32_t>({0, 2}, {2})};
  auto map_shallow = map;
  auto map_deep = map.deep_copy();
  map_shallow.f[0] = 5;
  map_deep.kept_ids[0] = 9;
  CHECK(map.f[0] == 5);
  CHECK(map.kept_ids[0] == 0);
}
TEST_CASE("a mesh keeps every structure it is given", "[cpp][core][handles]") {
  // a tag holds views into a structure's arrays and the cache REPLACES the
  // whole of one when it rebuilds it, so a mesh retains what it was handed
  const auto owned = handles_triangle();
  const auto mesh = owned.mesh();

  const auto link = mesh.vertex_link();
  const auto membership = mesh.face_membership();
  REQUIRE(link.size() == 3);
  REQUIRE(membership.size() == 3);
  CHECK(link[0][0] == 2);

  owned.cache.set_vertex_link(
      make_blocks<std::int32_t>({0, 2, 4, 6}, {1, 2, 2, 0, 0, 1}),
      mesh.geometry());
  owned.cache.set_face_membership(
      make_blocks<std::int32_t>({0, 1, 2, 3}, {0, 0, 0}), mesh.geometry());

  CHECK(link[0][0] == 2);
  CHECK(link.size() == 3);
  CHECK(membership.size() == 3);
  CHECK(owned.mesh().vertex_link()[0][0] == 1);
}

TEST_CASE("a stated structure is answered, and builds nothing",
          "[cpp][core][handles][cache-authority]") {
  const auto owned = handles_triangle();
  const auto mesh = owned.mesh();
  owned.cache.set_vertex_link(
      make_blocks<std::int32_t>({0, 2, 4, 6}, {1, 2, 0, 2, 0, 1}),
      mesh.geometry());
  REQUIRE(owned.cache.is_vertex_link_fresh(mesh.geometry()));
  REQUIRE_FALSE(owned.cache.is_face_membership_built());

  const auto reader = owned.mesh();
  const auto observed = reader.vertex_link();
  CHECK(observed.size() == 3);
  CHECK(observed[0][0] == 1);
  CHECK(observed[2][1] == 1);
  CHECK(owned.cache.vertex_link_build_count() == 0);
  CHECK(owned.cache.face_membership_build_count() == 0);
  CHECK_FALSE(owned.cache.is_face_membership_built());
}

TEST_CASE("a point cloud asks for its tree, and keeps the one it is given",
          "[cpp][core][handles]") {
  auto owned = tf::cpp::test::owned_point_cloud<float>{
      tf::cpp::test::points_of<float>({0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0})};

  // ASSEMBLY BUILDS NOTHING: the tree is asked for the first time a body wants
  // one, and retained from then on
  const auto cloud = owned.point_cloud();
  CHECK_FALSE(owned.cache.is_tree_built());
  CHECK(owned.cache.tree_build_count() == 0);

  CHECK(cloud.tree().bv().max[0] == 1);
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(cloud.form().tree().bv().max[0] == 1);
  CHECK(owned.cache.tree_build_count() == 1);

  auto &coordinates = owned.points.data_buffer();
  for (std::size_t i = 0; i != coordinates.size(); i += 3)
    coordinates[i] += 20;
  owned.cache.points_changed();
  const auto moved = owned.point_cloud();
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(moved.tree().bv().max[0] == 21);
  CHECK(owned.cache.tree_build_count() == 2);
  // the first cloud keeps the tree it was handed, whatever the cache holds now
  CHECK(cloud.tree().bv().max[0] == 1);
}
