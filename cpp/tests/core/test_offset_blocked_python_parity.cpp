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
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
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

using offset_blocks =
    tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>;

template <typename T>
auto check_uniform_conversion(std::initializer_list<T> values,
                              tf::small_vector<int, 3> shape,
                              std::initializer_list<T> expected_offsets)
    -> void {
  auto source = make_array<T>(values, std::move(shape));
  const auto blocks =
      tf::cpp::offset_blocked_buffer<T, T>::from_uniform(source);

  static_assert(
      std::is_same<decltype(blocks.offsets()), tf::cpp::nd_array<T>>::value,
      "uniform conversion preserves the offset type");
  static_assert(
      std::is_same<decltype(blocks.data()), tf::cpp::nd_array<T>>::value,
      "uniform conversion preserves the value type");
  REQUIRE(blocks.offsets().length() == expected_offsets.size());
  REQUIRE(blocks.data().length() == values.size());

  std::size_t index = 0;
  for (const auto expected : expected_offsets)
    CHECK(blocks.offsets()[index++] == expected);
  index = 0;
  for (const auto expected : values)
    CHECK(blocks.data()[index++] == expected);
}

} // namespace

TEST_CASE("uniform arrays convert to exact typed offset blocks",
          "[cpp][core][python-parity][offset-blocked][uniform]") {
  check_uniform_conversion<std::int32_t>({0, 1, 2, 3, 4, 5}, {2, 3}, {0, 3, 6});
  check_uniform_conversion<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7}, {2, 4},
                                         {0, 4, 8});
  check_uniform_conversion<std::int32_t>({0, 1, 2, 3, 4}, {1, 5}, {0, 5});

  check_uniform_conversion<std::int64_t>({0, 1, 2, 3, 4, 5}, {2, 3}, {0, 3, 6});
  check_uniform_conversion<std::int64_t>({0, 1, 2, 3, 4, 5, 6, 7}, {2, 4},
                                         {0, 4, 8});
  check_uniform_conversion<std::int64_t>({0, 1, 2, 3, 4}, {1, 5}, {0, 5});
}

TEST_CASE("uniform conversion owns a sealed flat copy",
          "[cpp][core][python-parity][offset-blocked][uniform][ownership]") {
  auto source32 = make_array<std::int32_t>({0, 1, 2, 3, 4, 5, 6, 7}, {2, 4});
  auto source64 = make_array<std::int64_t>({8, 9, 10}, {1, 3});
  const auto blocks32 = offset_blocks::from_uniform(source32);
  const auto blocks64 =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::from_uniform(
          source64);

  REQUIRE(blocks32.data().raw_owner().get() != source32.raw_owner().get());
  REQUIRE(blocks64.data().raw_owner().get() != source64.raw_owner().get());
  CHECK((blocks32.data().raw_shape() == tf::small_vector<int, 3>{8}));
  CHECK((blocks64.data().raw_shape() == tf::small_vector<int, 3>{3}));
  source32[0] = 99;
  source32[7] = 77;
  source64[0] = 88;
  CHECK(blocks32.data()[0] == 0);
  CHECK(blocks32.data()[7] == 7);
  CHECK(blocks64.data()[0] == 8);
}

TEST_CASE("uniform conversion supports zero rows with nonzero width",
          "[cpp][core][python-parity][offset-blocked][uniform][empty]") {
  const auto empty32 =
      offset_blocks::from_uniform(make_array<std::int32_t>({}, {0, 4}));
  const auto empty64 =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::from_uniform(
          make_array<std::int64_t>({}, {0, 5}));

  CHECK(empty32.size() == 0);
  REQUIRE(empty32.offsets().length() == 1);
  CHECK(empty32.offsets()[0] == 0);
  CHECK(empty32.data().empty());
  CHECK(empty64.size() == 0);
  REQUIRE(empty64.offsets().length() == 1);
  CHECK(empty64.offsets()[0] == 0);
  CHECK(empty64.data().empty());
}

TEST_CASE("uniform conversion rejects invalid rank storage and offset counts",
          "[cpp][core][python-parity][offset-blocked][uniform][validation]") {
  CHECK_THROWS_AS(
      offset_blocks::from_uniform(make_array<std::int32_t>({0, 1, 2, 3}, {4})),
      std::invalid_argument);
  CHECK_THROWS_AS(offset_blocks::from_uniform(
                      make_array<std::int32_t>({0, 1, 2, 3}, {1, 2, 2})),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      offset_blocks::from_uniform(tf::cpp::nd_array<std::int32_t>{}),
      std::invalid_argument);

  const auto rows = std::numeric_limits<int>::max();
  CHECK_THROWS_AS(
      offset_blocks::from_uniform(make_array<std::int32_t>({}, {rows, 0})),
      std::length_error);
  CHECK_THROWS_AS(
      (tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::from_uniform(
          make_array<std::int64_t>({}, {rows, 0}))),
      std::length_error);
}

TEST_CASE("as_offset_blocked is the public conversion facade",
          "[cpp][core][python-parity][offset-blocked][uniform][facade]") {
  using blocks32 = tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>;
  using blocks64 = tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>;
  using facade32 = blocks32 (*)(const tf::cpp::nd_array<std::int32_t> &);
  using facade64 = blocks64 (*)(const tf::cpp::nd_array<std::int64_t> &);
  const auto convert32 = static_cast<facade32>(&tf::cpp::as_offset_blocked);
  const auto convert64 = static_cast<facade64>(&tf::cpp::as_offset_blocked);

  const auto source32 = make_array<std::int32_t>({0, 1, 2, 3}, {1, 4});
  const auto source64 = make_array<std::int64_t>({4, 5, 6, 7}, {1, 4});
  const auto facade_result32 = convert32(source32);
  const auto member_result32 = blocks32::from_uniform(source32);
  const auto facade_result64 = convert64(source64);
  const auto member_result64 = blocks64::from_uniform(source64);

  REQUIRE(facade_result32.offsets().length() ==
          member_result32.offsets().length());
  REQUIRE(facade_result32.data().length() == member_result32.data().length());
  REQUIRE(facade_result64.offsets().length() ==
          member_result64.offsets().length());
  REQUIRE(facade_result64.data().length() == member_result64.data().length());
  for (std::size_t index = 0; index < facade_result32.offsets().length();
       ++index)
    CHECK(facade_result32.offsets()[index] == member_result32.offsets()[index]);
  for (std::size_t index = 0; index < facade_result32.data().length(); ++index)
    CHECK(facade_result32.data()[index] == member_result32.data()[index]);
  for (std::size_t index = 0; index < facade_result64.offsets().length();
       ++index)
    CHECK(facade_result64.offsets()[index] == member_result64.offsets()[index]);
  for (std::size_t index = 0; index < facade_result64.data().length(); ++index)
    CHECK(facade_result64.data()[index] == member_result64.data()[index]);
}

TEST_CASE("offset-blocked storage preserves public offsets and block shape",
          "[cpp][core][python-parity][offset-blocked]") {
  const auto blocks = offset_blocks::create(
      make_array<std::int32_t>({0, 3, 3, 7}, {4}),
      make_array<std::int32_t>({0, 1, 2, 4, 5, 6, 7}, {7}));

  REQUIRE(blocks.is_valid());
  CHECK(blocks.size() == 3);
  CHECK((blocks.offsets().raw_shape() == tf::small_vector<int, 3>{4}));
  CHECK((blocks.data().raw_shape() == tf::small_vector<int, 3>{7}));
  CHECK(blocks.offsets()[0] == 0);
  CHECK(blocks.offsets()[1] == 3);
  CHECK(blocks.offsets()[2] == 3);
  CHECK(blocks.offsets()[3] == 7);
  CHECK(blocks.get(0).length() == 3);
  CHECK(blocks.get(0)[2] == 2);
  CHECK(blocks.get(1).empty());
  REQUIRE(blocks.get(2).length() == 4);
  CHECK(blocks.get(2)[0] == 4);
  CHECK(blocks.get(2)[3] == 7);

  const auto empty = offset_blocks::create(make_array<std::int32_t>({0}, {1}),
                                           make_array<std::int32_t>({}, {0}));
  CHECK(empty.is_valid());
  CHECK(empty.size() == 0);
  CHECK(empty.offsets().length() == 1);
  CHECK(empty.offsets()[0] == 0);
  CHECK(empty.data().empty());
}

TEST_CASE("offset-blocked storage has explicit shallow and deep ownership",
          "[cpp][core][python-parity][offset-blocked][ownership]") {
  auto offsets = make_array<std::int32_t>({0, 2, 5}, {3});
  auto data = make_array<std::int32_t>({4, 5, 7, 8, 9}, {5});
  auto blocks = offset_blocks::create(offsets, data);
  auto shallow = blocks.shallow_copy();
  auto deep = blocks.deep_copy();

  CHECK(shallow.raw_offsets().get() == blocks.raw_offsets().get());
  CHECK(shallow.raw_data().get() == blocks.raw_data().get());
  CHECK(deep.raw_offsets().get() != blocks.raw_offsets().get());
  CHECK(deep.raw_data().get() != blocks.raw_data().get());

  offsets.destroy();
  data.destroy();
  REQUIRE(blocks.is_valid());
  CHECK(blocks.get(1)[2] == 9);

  shallow.data()[0] = 40;
  deep.data()[1] = 50;
  CHECK(blocks.data()[0] == 40);
  CHECK(blocks.data()[1] == 5);
  CHECK(deep.data()[0] == 4);
  CHECK(deep.data()[1] == 50);

  blocks.destroy();
  REQUIRE(shallow.is_valid());
  CHECK(shallow.get(0)[0] == 40);
}

TEST_CASE("offset-blocked storage rejects malformed offsets and access",
          "[cpp][core][python-parity][offset-blocked][validation]") {
  CHECK_THROWS_AS(
      offset_blocks::create(make_array<std::int32_t>({0, 2, 1}, {3}),
                            make_array<std::int32_t>({4}, {1})),
      std::invalid_argument);
  CHECK_THROWS_AS(offset_blocks::create(make_array<std::int32_t>({0, 1}, {2}),
                                        make_array<std::int32_t>({4, 5}, {2})),
                  std::invalid_argument);
  CHECK_THROWS_AS(offset_blocks::create(make_array<std::int32_t>({}, {0}),
                                        make_array<std::int32_t>({4}, {1})),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      offset_blocks::create(make_array<std::int32_t>({0, 2}, {1, 2}),
                            make_array<std::int32_t>({4, 5}, {2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      offset_blocks::create(make_array<std::int32_t>({0, 2}, {2}),
                            make_array<std::int32_t>({4, 5}, {1, 2})),
      std::invalid_argument);

  const auto blocks =
      offset_blocks::create(make_array<std::int32_t>({0, 2}, {2}),
                            make_array<std::int32_t>({4, 5}, {2}));
  CHECK_THROWS_AS(blocks.get(-1), std::out_of_range);
  CHECK_THROWS_AS(blocks.get(1), std::out_of_range);

  const offset_blocks invalid;
  CHECK_FALSE(invalid.is_valid());
  CHECK(invalid.size() == 0);
  CHECK_THROWS_AS(invalid.get(0), std::logic_error);
}
