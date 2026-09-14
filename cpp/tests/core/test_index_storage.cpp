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
#include "trueform/cpp/core/index_map.hpp"
#include "trueform/cpp/core/common_index.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

template <typename> struct function_argument;

template <typename Result, typename Argument>
struct function_argument<Result (*)(Argument)> {
  using type = std::remove_reference_t<Argument>;
};

template <typename T>
auto make_array(std::initializer_list<T> values,
                std::initializer_list<int> shape) -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(values.size());
  auto output = storage.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           tf::small_vector<int, 3>(shape));
}

using int64_blocks = tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>;

constexpr auto large_identity =
    static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 42;

static_assert(std::is_same_v<tf::cpp::default_index_t, std::int32_t>);
static_assert(
    std::is_same_v<tf::cpp::index_map<>, tf::cpp::index_map<std::int32_t>>);
static_assert(std::is_same_v<decltype(std::declval<tf::cpp::index_map<>>().f),
                             tf::cpp::nd_array<std::int32_t>>);
static_assert(
    std::is_same_v<
        decltype(std::declval<tf::cpp::index_map<std::int64_t>>().kept_ids),
        tf::cpp::nd_array<std::int64_t>>);
static_assert(std::is_aggregate_v<tf::cpp::index_map<>>);
static_assert(std::is_aggregate_v<tf::cpp::index_map<std::int64_t>>);
static_assert(tf::cpp::is_supported_index_v<std::int32_t>);
static_assert(tf::cpp::is_supported_index_v<const std::int32_t>);
static_assert(tf::cpp::is_supported_index_v<std::int64_t>);
static_assert(!tf::cpp::is_supported_index_v<std::int8_t>);
static_assert(!tf::cpp::is_supported_index_v<std::uint32_t>);
static_assert(!tf::cpp::is_supported_index_v<std::uint64_t>);
static_assert(!tf::cpp::is_supported_index_v<float>);
static_assert(
    std::is_same_v<tf::cpp::common_index_t<std::int32_t>, std::int32_t>);
static_assert(
    std::is_same_v<tf::cpp::common_index_t<std::int64_t>, std::int64_t>);
static_assert(
    std::is_same_v<tf::cpp::common_index_t<std::int32_t, std::int64_t>,
                   std::int64_t>);
static_assert(
    std::is_same_v<tf::cpp::common_index_t<const std::int32_t, std::int32_t>,
                   std::int32_t>);

} // namespace

TEST_CASE("index traits expose the signed identity storage contract",
          "[cpp][core][index-storage]") {
  CHECK(std::is_signed_v<tf::cpp::default_index_t>);
  CHECK(sizeof(tf::cpp::common_index_t<std::int32_t, std::int64_t>) ==
        sizeof(std::int64_t));
}

TEST_CASE("typed index maps preserve source compatibility width and ownership",
          "[cpp][core][index-storage][index-map]") {
  tf::cpp::index_map<> source_compatible{
      make_array<std::int32_t>({0, 3, 1}, {3}),
      make_array<std::int32_t>({0, 2}, {2})};
  REQUIRE(source_compatible.is_valid());
  CHECK(source_compatible.f[1] == 3);
  CHECK(source_compatible.f[2] == 1);
  CHECK(source_compatible.kept_ids[1] == 2);

  using int64_map = tf::cpp::index_map<std::int64_t>;
  using core_buffer = typename function_argument<
      decltype(&int64_map::from_index_map_buffer)>::type;
  core_buffer buffer;
  buffer.f().allocate(3);
  buffer.f()[0] = 0;
  buffer.f()[1] = 3;
  buffer.f()[2] = 1;
  buffer.kept_ids().allocate(2);
  buffer.kept_ids()[0] = 0;
  buffer.kept_ids()[1] = 2;

  auto map = tf::cpp::index_map<std::int64_t>::from_index_map_buffer(
      std::move(buffer));
  REQUIRE(map.is_valid());
  CHECK(map.f[1] == std::int64_t{3});
  CHECK(map.f[2] == std::int64_t{1});
  CHECK(map.kept_ids[0] == std::int64_t{0});
  CHECK(map.kept_ids[1] == std::int64_t{2});

  auto shallow = map;
  auto deep = map.deep_copy();
  auto retained_f = map.f;
  REQUIRE(shallow.f.raw_owner().get() == map.f.raw_owner().get());
  REQUIRE(shallow.kept_ids.raw_owner().get() == map.kept_ids.raw_owner().get());
  REQUIRE(deep.f.raw_owner().get() != map.f.raw_owner().get());
  REQUIRE(deep.kept_ids.raw_owner().get() != map.kept_ids.raw_owner().get());

  CHECK(shallow.f[1] == std::int64_t{3});
  CHECK(deep.f[1] == std::int64_t{3});
  shallow.f[0] = 1;
  deep.kept_ids[0] = 1;
  CHECK(map.f[0] == std::int64_t{1});
  CHECK(map.kept_ids[0] == std::int64_t{0});
  CHECK(deep.kept_ids[0] == std::int64_t{1});

  map = {};
  CHECK_FALSE(map.is_valid());
  REQUIRE(shallow.is_valid());
  REQUIRE(retained_f.is_valid());
  CHECK(retained_f[0] == std::int64_t{1});

  shallow = {};
  CHECK_FALSE(shallow.is_valid());
  CHECK(retained_f[0] == std::int64_t{1});
  CHECK(deep.kept_ids[0] == std::int64_t{1});

  int64_map raw_identity_transport{
      make_array<std::int64_t>({0, large_identity}, {2}),
      make_array<std::int64_t>({large_identity + 1}, {1})};
  REQUIRE(raw_identity_transport.is_valid());
  CHECK(raw_identity_transport.f[1] == large_identity);
  CHECK(raw_identity_transport.kept_ids[0] == large_identity + 1);
  const auto detached_transport = raw_identity_transport.deep_copy();
  raw_identity_transport = {};
  CHECK(detached_transport.f[1] == large_identity);
  CHECK(detached_transport.kept_ids[0] == large_identity + 1);
}

TEST_CASE("int64 arrays preserve views ownership and large identities",
          "[cpp][core][index-storage][nd-array]") {
  auto array = make_array<std::int64_t>(
      {0, large_identity, large_identity + 1, 3, 4, large_identity + 2},
      {2, 3});

  REQUIRE(array.is_valid());
  CHECK(array.size() == 6);
  CHECK(array.length() == 6);
  CHECK(array.ndim() == 2);
  CHECK(array.shape_at(0) == 2);
  CHECK(array.shape_at(1) == 3);
  CHECK(array[1] == large_identity);

  auto row = array.row(1);
  auto slice = array.slice(0, 1);
  auto reshaped = array.reshape({3, 2});
  CHECK(row.length() == 3);
  CHECK(row[2] == large_identity + 2);
  CHECK(slice.shape_at(0) == 1);
  CHECK(slice[1] == large_identity);
  CHECK(reshaped.shape_at(0) == 3);
  CHECK(reshaped.shape_at(1) == 2);

  auto shallow = array.shallow_copy();
  auto deep = array.deep_copy();
  REQUIRE(shallow.raw_owner().get() == array.raw_owner().get());
  REQUIRE(deep.raw_owner().get() != array.raw_owner().get());
  shallow[0] = large_identity + 3;
  deep[1] = large_identity + 4;
  CHECK(array[0] == large_identity + 3);
  CHECK(array[1] == large_identity);
  CHECK(deep[1] == large_identity + 4);

  array.destroy();
  CHECK_FALSE(array.is_valid());
  CHECK(array.length() == 0);
  REQUIRE(shallow.is_valid());
  REQUIRE(row.is_valid());
  CHECK(shallow[0] == large_identity + 3);
  CHECK(row[2] == large_identity + 2);

  row.destroy();
  CHECK_FALSE(row.is_valid());
}

TEST_CASE("int64 offset blocks validate access and explicit ownership",
          "[cpp][core][index-storage][offset-blocked]") {
  auto offsets = make_array<std::int64_t>({0, 2, 2, 5}, {4});
  auto identities = make_array<std::int64_t>(
      {large_identity, large_identity + 1, large_identity + 2,
       large_identity + 3, large_identity + 4},
      {5});
  auto blocks = int64_blocks::create(offsets, identities);

  REQUIRE(blocks.is_valid());
  CHECK(blocks.size() == 3);
  REQUIRE(blocks.get(0).length() == 2);
  CHECK(blocks.get(0)[1] == large_identity + 1);
  CHECK(blocks.get(1).empty());
  REQUIRE(blocks.get(2).length() == 3);
  CHECK(blocks.get(2)[0] == large_identity + 2);
  CHECK(blocks.get(2)[2] == large_identity + 4);

  auto shallow = blocks.shallow_copy();
  auto deep = blocks.deep_copy();
  CHECK(shallow.raw_offsets().get() == blocks.raw_offsets().get());
  CHECK(shallow.raw_data().get() == blocks.raw_data().get());
  CHECK(deep.raw_offsets().get() != blocks.raw_offsets().get());
  CHECK(deep.raw_data().get() != blocks.raw_data().get());

  shallow.data()[0] = large_identity + 10;
  deep.data()[1] = large_identity + 11;
  CHECK(blocks.data()[0] == large_identity + 10);
  CHECK(blocks.data()[1] == large_identity + 1);
  CHECK(deep.data()[1] == large_identity + 11);

  auto retained_block = blocks.get(2);
  blocks.destroy();
  CHECK_FALSE(blocks.is_valid());
  REQUIRE(shallow.is_valid());
  CHECK(shallow.get(0)[0] == large_identity + 10);
  CHECK(retained_block[2] == large_identity + 4);
}

TEST_CASE("int64 offset blocks reject malformed offsets",
          "[cpp][core][index-storage][offset-blocked][validation]") {
  CHECK_THROWS_AS(int64_blocks::create(make_array<std::int64_t>({1, 2}, {2}),
                                       make_array<std::int64_t>({4, 5}, {2})),
                  std::invalid_argument);
  CHECK_THROWS_AS(int64_blocks::create(make_array<std::int64_t>({0, 2, 1}, {3}),
                                       make_array<std::int64_t>({4}, {1})),
                  std::invalid_argument);
  CHECK_THROWS_AS(int64_blocks::create(make_array<std::int64_t>({0, 1}, {2}),
                                       make_array<std::int64_t>({4, 5}, {2})),
                  std::invalid_argument);
  CHECK_THROWS_AS(int64_blocks::create(
                      make_array<std::int64_t>(
                          {0, static_cast<std::int64_t>(
                                  std::numeric_limits<std::uint32_t>::max()) +
                                  2},
                          {2}),
                      make_array<std::int64_t>({4}, {1})),
                  std::invalid_argument);
  CHECK_THROWS_AS(int64_blocks::create(make_array<std::int64_t>({}, {0}),
                                       make_array<std::int64_t>({4}, {1})),
                  std::invalid_argument);
  CHECK_THROWS_AS(int64_blocks::create(make_array<std::int64_t>({0, 2}, {1, 2}),
                                       make_array<std::int64_t>({4, 5}, {2})),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      int64_blocks::create(make_array<std::int64_t>({0, 2}, {2}),
                           make_array<std::int64_t>({4, 5}, {1, 2})),
      std::invalid_argument);

  const auto blocks = int64_blocks::create(
      make_array<std::int64_t>({0, 2}, {2}),
      make_array<std::int64_t>({large_identity, large_identity + 1}, {2}));
  CHECK_THROWS_AS(blocks.get(-1), std::out_of_range);
  CHECK_THROWS_AS(blocks.get(1), std::out_of_range);

  const int64_blocks invalid;
  CHECK_FALSE(invalid.is_valid());
  CHECK(invalid.size() == 0);
  CHECK_THROWS_AS(invalid.get(0), std::logic_error);
}
