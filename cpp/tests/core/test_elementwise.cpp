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
#include "trueform/cpp/core/elementwise.hpp"
#include "trueform/cpp/core/elementwise/detail/broadcast.hpp"
#include "trueform/cpp/core/parallel_config.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <tbb/global_control.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <initializer_list>
#include <memory>
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

template <typename T>
auto check_values(const tf::cpp::nd_array<T> &array,
                  std::initializer_list<T> expected) -> void {
  REQUIRE(array.length() == expected.size());
  auto index = std::size_t{0};
  for (const auto value : expected)
    CHECK(array[index++] == value);
}

template <typename T, typename Generator>
auto make_generated_array(tf::small_vector<int, 3> shape, Generator generator)
    -> tf::cpp::nd_array<T> {
  const auto length = static_cast<std::size_t>(tf::cpp::detail::total_size(shape));
  tf::buffer<T> buffer;
  buffer.allocate(length);
  for (std::size_t index = 0; index < length; ++index)
    buffer[index] = generator(index);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T, typename Operation>
auto check_old_index_oracle(const tf::cpp::nd_array<T> &a,
                            const tf::cpp::nd_array<T> &b,
                            const tf::cpp::nd_array<T> &result,
                            Operation operation) -> void {
  const auto output_shape =
      tf::cpp::detail::broadcast_shape(a.raw_shape(), b.raw_shape());
  const auto output_strides = tf::cpp::detail::compute_strides(output_shape);
  const auto a_strides =
      tf::cpp::detail::broadcast_strides(a.raw_shape(), output_shape);
  const auto b_strides =
      tf::cpp::detail::broadcast_strides(b.raw_shape(), output_shape);
  const auto dimensions = static_cast<int>(output_shape.size());
  const auto total = tf::cpp::detail::total_size(output_shape);
  REQUIRE(result.raw_shape() == output_shape);
  REQUIRE(result.length() == static_cast<std::size_t>(total));
  for (auto index = 0; index < total; ++index) {
    const auto a_index =
        tf::cpp::detail::broadcast_index(index, output_strides, a_strides, dimensions);
    const auto b_index =
        tf::cpp::detail::broadcast_index(index, output_strides, b_strides, dimensions);
    CHECK(result[static_cast<std::size_t>(index)] ==
          operation(a[static_cast<std::size_t>(a_index)],
                    b[static_cast<std::size_t>(b_index)]));
  }
}

template <typename T>
auto check_assignment_old_index_oracle(const tf::cpp::nd_array<T> &target,
                                       const tf::cpp::nd_array<T> &source)
    -> void {
  const auto output_shape =
      tf::cpp::detail::broadcast_shape(target.raw_shape(), source.raw_shape());
  const auto output_strides = tf::cpp::detail::compute_strides(output_shape);
  const auto source_strides =
      tf::cpp::detail::broadcast_strides(source.raw_shape(), output_shape);
  const auto dimensions = static_cast<int>(output_shape.size());
  REQUIRE(target.raw_shape() == output_shape);
  for (auto index = 0; index < tf::cpp::detail::total_size(output_shape); ++index) {
    const auto source_index = tf::cpp::detail::broadcast_index(
        index, output_strides, source_strides, dimensions);
    CHECK(target[static_cast<std::size_t>(index)] ==
          source[static_cast<std::size_t>(source_index)]);
  }
}

template <typename T>
auto check_same_storage(const tf::cpp::nd_array<T> &a,
                        const tf::cpp::nd_array<T> &b) -> void {
  REQUIRE(a.raw_shape() == b.raw_shape());
  REQUIRE(a.length() == b.length());
  CHECK(std::memcmp(a.raw_data(), b.raw_data(), a.length() * sizeof(T)) == 0);
}

template <typename Operation>
auto run_with_thread_cap(std::size_t thread_cap, Operation operation)
    -> decltype(operation()) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                              thread_cap);
  return operation();
}

template <typename T> auto check_arithmetic() -> void {
  const auto a = make_array<T>({T{12}, T{18}, T{24}, T{30}}, {2, 2});
  const auto b = make_array<T>({T{3}, T{3}}, {2});

  check_values(tf::cpp::add(a, b), {T{15}, T{21}, T{27}, T{33}});
  check_values(tf::cpp::sub(a, b), {T{9}, T{15}, T{21}, T{27}});
  check_values(tf::cpp::mul(a, b), {T{36}, T{54}, T{72}, T{90}});
  check_values(tf::cpp::div(a, b), {T{4}, T{6}, T{8}, T{10}});
  check_values(tf::cpp::mod(a, b), {T{0}, T{0}, T{0}, T{0}});

  check_values(tf::cpp::add_scalar(a, T{2}), {T{14}, T{20}, T{26}, T{32}});
  check_values(tf::cpp::sub_scalar(a, T{2}), {T{10}, T{16}, T{22}, T{28}});
  check_values(tf::cpp::mul_scalar(a, T{2}), {T{24}, T{36}, T{48}, T{60}});
  check_values(tf::cpp::div_scalar(a, T{2}), {T{6}, T{9}, T{12}, T{15}});
  check_values(tf::cpp::mod_scalar(a, T{7}), {T{5}, T{4}, T{3}, T{2}});

  auto result = a.deep_copy();
  tf::cpp::add_inplace(result, b);
  check_values(result, {T{15}, T{21}, T{27}, T{33}});
  result = a.deep_copy();
  tf::cpp::sub_inplace(result, b);
  check_values(result, {T{9}, T{15}, T{21}, T{27}});
  result = a.deep_copy();
  tf::cpp::mul_inplace(result, b);
  check_values(result, {T{36}, T{54}, T{72}, T{90}});
  result = a.deep_copy();
  tf::cpp::div_inplace(result, b);
  check_values(result, {T{4}, T{6}, T{8}, T{10}});
  result = a.deep_copy();
  tf::cpp::mod_inplace(result, b);
  check_values(result, {T{0}, T{0}, T{0}, T{0}});

  result = a.deep_copy();
  tf::cpp::add_scalar_inplace(result, T{2});
  check_values(result, {T{14}, T{20}, T{26}, T{32}});
  result = a.deep_copy();
  tf::cpp::sub_scalar_inplace(result, T{2});
  check_values(result, {T{10}, T{16}, T{22}, T{28}});
  result = a.deep_copy();
  tf::cpp::mul_scalar_inplace(result, T{2});
  check_values(result, {T{24}, T{36}, T{48}, T{60}});
  result = a.deep_copy();
  tf::cpp::div_scalar_inplace(result, T{2});
  check_values(result, {T{6}, T{9}, T{12}, T{15}});
  result = a.deep_copy();
  tf::cpp::mod_scalar_inplace(result, T{7});
  check_values(result, {T{5}, T{4}, T{3}, T{2}});
}

template <typename T> auto check_assignments() -> void {
  auto target = make_array<T>({T{0}, T{0}, T{0}, T{0}, T{0}, T{0}}, {3, 2});
  tf::cpp::assign_scalar(target, T{1});
  check_values(target, {T{1}, T{1}, T{1}, T{1}, T{1}, T{1}});

  const auto row = make_array<T>({T{2}, T{3}}, {2});
  tf::cpp::assign_array(target, row);
  check_values(target, {T{2}, T{3}, T{2}, T{3}, T{2}, T{3}});

  const auto middle = make_array<std::int32_t>({1}, {1});
  tf::cpp::assign_indexed_scalar(target, middle, T{9});
  check_values(target, {T{2}, T{3}, T{9}, T{9}, T{2}, T{3}});

  const auto outside = make_array<std::int32_t>({0, 2}, {2});
  const auto indexed_values = make_array<T>({T{4}, T{5}}, {2});
  tf::cpp::assign_indexed_array(target, outside, indexed_values);
  check_values(target, {T{4}, T{5}, T{9}, T{9}, T{4}, T{5}});

  const auto middle_mask = make_array<std::int8_t>(
      {std::int8_t{0}, std::int8_t{1}, std::int8_t{0}}, {3});
  tf::cpp::assign_masked_scalar(target, middle_mask, T{6});
  check_values(target, {T{4}, T{5}, T{6}, T{6}, T{4}, T{5}});

  const auto outside_mask = make_array<std::int8_t>(
      {std::int8_t{1}, std::int8_t{0}, std::int8_t{1}}, {3});
  const auto masked_values = make_array<T>({T{7}, T{8}}, {2});
  tf::cpp::assign_masked_array(target, outside_mask, masked_values);
  check_values(target, {T{7}, T{8}, T{6}, T{6}, T{7}, T{8}});
}

template <typename T> auto check_comparisons() -> void {
  const auto a = make_array<T>({T{1}, T{3}, T{2}, T{4}}, {2, 2});
  const auto b = make_array<T>({T{1}, T{2}}, {2});
  check_values(tf::cpp::eq(a, b), {std::int8_t{1}, std::int8_t{0},
                                   std::int8_t{0}, std::int8_t{0}});
  check_values(tf::cpp::neq(a, b), {std::int8_t{0}, std::int8_t{1},
                                    std::int8_t{1}, std::int8_t{1}});
  check_values(tf::cpp::lt(a, b), {std::int8_t{0}, std::int8_t{0},
                                   std::int8_t{0}, std::int8_t{0}});
  check_values(tf::cpp::gt(a, b), {std::int8_t{0}, std::int8_t{1},
                                   std::int8_t{1}, std::int8_t{1}});
  check_values(tf::cpp::lte(a, b), {std::int8_t{1}, std::int8_t{0},
                                    std::int8_t{0}, std::int8_t{0}});
  check_values(tf::cpp::gte(a, b), {std::int8_t{1}, std::int8_t{1},
                                    std::int8_t{1}, std::int8_t{1}});

  check_values(tf::cpp::eq_scalar(a, T{2}), {std::int8_t{0}, std::int8_t{0},
                                             std::int8_t{1}, std::int8_t{0}});
  check_values(tf::cpp::neq_scalar(a, T{2}), {std::int8_t{1}, std::int8_t{1},
                                              std::int8_t{0}, std::int8_t{1}});
  check_values(tf::cpp::lt_scalar(a, T{3}), {std::int8_t{1}, std::int8_t{0},
                                             std::int8_t{1}, std::int8_t{0}});
  check_values(tf::cpp::gt_scalar(a, T{2}), {std::int8_t{0}, std::int8_t{1},
                                             std::int8_t{0}, std::int8_t{1}});
  check_values(tf::cpp::lte_scalar(a, T{2}), {std::int8_t{1}, std::int8_t{0},
                                              std::int8_t{1}, std::int8_t{0}});
  check_values(tf::cpp::gte_scalar(a, T{3}), {std::int8_t{0}, std::int8_t{1},
                                              std::int8_t{0}, std::int8_t{1}});
}

template <typename T, typename CopyOperation, typename InplaceOperation>
auto check_float_unary(const tf::cpp::nd_array<T> &input, T expected,
                       CopyOperation copy_operation,
                       InplaceOperation inplace_operation) -> void {
  const auto result = copy_operation(input);
  CHECK(result[0] == Catch::Approx(expected).epsilon(1e-5));
  auto inplace = input.deep_copy();
  inplace_operation(inplace);
  CHECK(inplace[0] == Catch::Approx(expected).epsilon(1e-5));
}

template <typename T> auto check_float_unaries() -> void {
  check_float_unary(
      make_array<T>({T{0.25}}, {1}), T{0.5},
      [](const auto &a) { return tf::cpp::sqrt(a); },
      [](auto &a) { tf::cpp::sqrt_inplace(a); });
  check_float_unary(
      make_array<T>({T{0.5}}, {1}), static_cast<T>(std::sin(0.5)),
      [](const auto &a) { return tf::cpp::sin(a); },
      [](auto &a) { tf::cpp::sin_inplace(a); });
  check_float_unary(
      make_array<T>({T{0.5}}, {1}), static_cast<T>(std::cos(0.5)),
      [](const auto &a) { return tf::cpp::cos(a); },
      [](auto &a) { tf::cpp::cos_inplace(a); });
  check_float_unary(
      make_array<T>({T{0.5}}, {1}), static_cast<T>(std::tan(0.5)),
      [](const auto &a) { return tf::cpp::tan(a); },
      [](auto &a) { tf::cpp::tan_inplace(a); });
  check_float_unary(
      make_array<T>({T{0.5}}, {1}), static_cast<T>(std::asin(0.5)),
      [](const auto &a) { return tf::cpp::asin(a); },
      [](auto &a) { tf::cpp::asin_inplace(a); });
  check_float_unary(
      make_array<T>({T{0.5}}, {1}), static_cast<T>(std::acos(0.5)),
      [](const auto &a) { return tf::cpp::acos(a); },
      [](auto &a) { tf::cpp::acos_inplace(a); });
  check_float_unary(
      make_array<T>({T{0.5}}, {1}), static_cast<T>(std::atan(0.5)),
      [](const auto &a) { return tf::cpp::atan(a); },
      [](auto &a) { tf::cpp::atan_inplace(a); });
  check_float_unary(
      make_array<T>({T{0.5}}, {1}), static_cast<T>(std::exp(0.5)),
      [](const auto &a) { return tf::cpp::exp(a); },
      [](auto &a) { tf::cpp::exp_inplace(a); });
  check_float_unary(
      make_array<T>({T{2}}, {1}), static_cast<T>(std::log(2.0)),
      [](const auto &a) { return tf::cpp::log(a); },
      [](auto &a) { tf::cpp::log_inplace(a); });
  check_float_unary(
      make_array<T>({T{8}}, {1}), T{3},
      [](const auto &a) { return tf::cpp::log2(a); },
      [](auto &a) { tf::cpp::log2_inplace(a); });
  check_float_unary(
      make_array<T>({T{100}}, {1}), T{2},
      [](const auto &a) { return tf::cpp::log10(a); },
      [](auto &a) { tf::cpp::log10_inplace(a); });
  check_float_unary(
      make_array<T>({T{1.75}}, {1}), T{1},
      [](const auto &a) { return tf::cpp::floor(a); },
      [](auto &a) { tf::cpp::floor_inplace(a); });
  check_float_unary(
      make_array<T>({T{1.25}}, {1}), T{2},
      [](const auto &a) { return tf::cpp::ceil(a); },
      [](auto &a) { tf::cpp::ceil_inplace(a); });
  check_float_unary(
      make_array<T>({T{1.5}}, {1}), T{2},
      [](const auto &a) { return tf::cpp::round(a); },
      [](auto &a) { tf::cpp::round_inplace(a); });
}

template <typename T> auto check_general_unaries() -> void {
  const auto input = make_array<T>({T{-3}, T{1}, T{5}}, {3});
  check_values(tf::cpp::abs(input), {T{3}, T{1}, T{5}});
  check_values(tf::cpp::neg(input), {T{3}, T{-1}, T{-5}});
  check_values(tf::cpp::clip(input, T{-2}, T{2}), {T{-2}, T{1}, T{2}});

  auto result = input.deep_copy();
  tf::cpp::abs_inplace(result);
  check_values(result, {T{3}, T{1}, T{5}});
  result = input.deep_copy();
  tf::cpp::neg_inplace(result);
  check_values(result, {T{3}, T{-1}, T{-5}});
  result = input.deep_copy();
  tf::cpp::clip_inplace(result, T{-2}, T{2});
  check_values(result, {T{-2}, T{1}, T{2}});
}

template <typename T> auto check_vector_operations() -> void {
  const auto a = make_array<T>({T{1}, T{2}, T{3}, T{4}, T{5}, T{6}}, {2, 3});
  const auto b = make_array<T>({T{6}, T{5}, T{4}, T{3}, T{2}, T{1}}, {2, 3});
  check_values(tf::cpp::dot(a, b), {T{28}, T{28}});
  check_values(tf::cpp::cross(a, b),
               {T{-7}, T{14}, T{-7}, T{-7}, T{14}, T{-7}});

  const auto matrix_a = make_array<T>({T{1}, T{2}, T{3}, T{4}}, {2, 2});
  const auto matrix_b = make_array<T>({T{5}, T{6}, T{7}, T{8}}, {2, 2});
  check_values(tf::cpp::mat_mul(matrix_a, matrix_b),
               {T{19}, T{22}, T{43}, T{50}});
}

template <typename T> auto check_singleton_batch_mat_mul() -> void {
  const auto singleton_left =
      make_array<T>({T{1}, T{2}, T{3}, T{4}}, {1, 2, 2});
  const auto batched_right = make_array<T>(
      {T{1}, T{0}, T{0}, T{1}, T{2}, T{0}, T{0}, T{2}}, {2, 2, 2});
  const auto left_result = tf::cpp::mat_mul(singleton_left, batched_right);
  CHECK((left_result.raw_shape() == tf::small_vector<int, 3>{2, 2, 2}));
  check_values(left_result, {T{1}, T{2}, T{3}, T{4}, T{2}, T{4}, T{6}, T{8}});

  const auto batched_left = make_array<T>(
      {T{1}, T{0}, T{0}, T{1}, T{2}, T{0}, T{0}, T{2}}, {2, 2, 2});
  const auto singleton_right =
      make_array<T>({T{5}, T{6}, T{7}, T{8}}, {1, 2, 2});
  const auto right_result = tf::cpp::mat_mul(batched_left, singleton_right);
  CHECK((right_result.raw_shape() == tf::small_vector<int, 3>{2, 2, 2}));
  check_values(right_result,
               {T{5}, T{6}, T{7}, T{8}, T{10}, T{12}, T{14}, T{16}});
}

template <typename From, typename To> auto check_cast() -> void {
  const auto input = make_array<From>({From{1}, From{2}}, {2});
  const auto output = tf::cpp::cast<From, To>(input);
  check_values(output, {To{1}, To{2}});
}

template <typename T> struct destroying_resolver {
  int *submissions;
  tf::cpp::nd_array<T> *input;

  template <typename Result>
  using state_type = tf::cpp::async::detail::future_state<Result>;

  template <typename Result>
  auto make_state() const -> std::shared_ptr<state_type<Result>> {
    ++*submissions;
    input->destroy();
    return std::make_shared<state_type<Result>>();
  }
};

template <typename T>
auto check_inverted_result(const tf::cpp::nd_array<T> &result, int size,
                           std::initializer_list<T> expected) -> void {
  CHECK(result.ndim() == 2);
  CHECK(result.shape_at(0) == size);
  CHECK(result.shape_at(1) == size);
  REQUIRE(result.length() == expected.size());
  auto index = std::size_t{0};
  for (const auto value : expected)
    CHECK(result[index++] == Catch::Approx(value).epsilon(1e-5));
}

template <typename T> auto check_inverted_facades() -> void {
  const auto matrix_3x3 = make_array<T>(
      {T{2}, T{0}, T{4}, T{0}, T{4}, T{8}, T{7}, T{8}, T{9}}, {3, 3});
  const auto expected_3x3 = std::initializer_list<T>{
      T{0.5}, T{0}, T{-2}, T{0}, T{0.25}, T{-2}, T{0}, T{0}, T{1}};
  check_inverted_result(tf::cpp::inverted(matrix_3x3), 3, expected_3x3);
  check_inverted_result(tf::cpp::async::inverted(matrix_3x3).get(), 3,
                        expected_3x3);

  const auto matrix_4x4 =
      make_array<T>({T{2}, T{0}, T{0}, T{6}, T{0}, T{4}, T{0}, T{8}, T{0}, T{0},
                     T{5}, T{10}, T{3}, T{4}, T{5}, T{6}},
                    {4, 4});
  const auto expected_4x4 = std::initializer_list<T>{
      T{0.5}, T{0}, T{0},   T{-3}, T{0}, T{0.25}, T{0}, T{-2},
      T{0},   T{0}, T{0.2}, T{-2}, T{0}, T{0},    T{0}, T{1}};
  check_inverted_result(tf::cpp::inverted(matrix_4x4), 4, expected_4x4);
  check_inverted_result(tf::cpp::async::inverted(matrix_4x4).get(), 4,
                        expected_4x4);

  for (const auto *matrix : {&matrix_3x3, &matrix_4x4}) {
    auto pending =
        tf::cpp::async::inverted(tf::cpp::async::future_resolver{}, *matrix);
    static_assert(
        std::is_same_v<decltype(pending), std::future<tf::cpp::nd_array<T>>>);
    const auto custom = pending.get();
    const auto size = matrix->shape_at(0);
    check_inverted_result(custom, size,
                          size == 3 ? expected_3x3 : expected_4x4);
  }
}

using float_array = tf::cpp::nd_array<float>;

static_assert(std::is_same<decltype(tf::cpp::async::add(
                               std::declval<const float_array &>(),
                               std::declval<const float_array &>())),
                           std::future<float_array>>::value,
              "default async array arithmetic preserves its exact future type");
static_assert(
    std::is_same<decltype(tf::cpp::async::mul_scalar(
                     std::declval<const float_array &>(), 2.0F)),
                 std::future<float_array>>::value,
    "default async scalar arithmetic preserves its exact future type");
static_assert(std::is_same<decltype(tf::cpp::async::gte_scalar(
                               std::declval<const float_array &>(), 2.0F)),
                           std::future<tf::cpp::nd_array<std::int8_t>>>::value,
              "default async comparisons expose their result dtype");
static_assert(std::is_same<decltype(tf::cpp::async::clip_inplace(
                               std::declval<float_array &>(), 0.0F, 1.0F)),
                           std::future<void>>::value,
              "default async in-place operations resolve void");
static_assert(std::is_same<decltype(tf::cpp::async::inverted(
                               std::declval<const float_array &>())),
                           std::future<float_array>>::value,
              "default async inversion preserves its exact future type");

} // namespace

TEST_CASE("broadcast helpers expose native shape and index calculation",
          "[cpp][core][elementwise]") {
  const auto shape = tf::cpp::detail::broadcast_shape({2, 1, 3}, {1, 4, 1});
  CHECK((shape == tf::small_vector<int, 3>{2, 4, 3}));
  CHECK(
      (tf::cpp::detail::compute_strides(shape) == tf::small_vector<int, 3>{12, 3, 1}));
  CHECK((tf::cpp::detail::broadcast_strides({1, 4, 1}, shape) ==
         tf::small_vector<int, 3>{0, 1, 0}));
  CHECK(tf::cpp::detail::broadcast_index(19, {12, 3, 1}, {0, 1, 0}, 3) == 2);
  CHECK(tf::cpp::detail::shapes_equal(shape, {2, 4, 3}));
  CHECK(tf::cpp::detail::total_size(shape) == 24);
}

TEST_CASE(
    "broadcast arithmetic preserves flat order for direct and vec3 operands",
    "[cpp][core][elementwise][broadcast]") {
  const auto matrix = make_array<std::int32_t>(
      {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}, {4, 3});
  const auto vector = make_array<std::int32_t>({1, 2, 3}, {3});

  const auto matrix_vector = tf::cpp::sub(matrix, vector);
  const auto vector_matrix = tf::cpp::sub(vector, matrix);
  check_old_index_oracle(matrix, vector, matrix_vector,
                         [](auto x, auto y) { return x - y; });
  check_old_index_oracle(vector, matrix, vector_matrix,
                         [](auto x, auto y) { return x - y; });
  check_values(matrix_vector,
               {9, 18, 27, 39, 48, 57, 69, 78, 87, 99, 108, 117});
  check_values(vector_matrix,
               {-9, -18, -27, -39, -48, -57, -69, -78, -87, -99, -108, -117});

  auto inplace = matrix.deep_copy();
  tf::cpp::add_inplace(inplace, vector);
  check_values(inplace, {11, 22, 33, 41, 52, 63, 71, 82, 93, 101, 112, 123});
}

TEST_CASE("broadcast old-index oracle covers every planned mapping",
          "[cpp][core][elementwise][broadcast]") {
  const auto check_both_orders = [](const auto &a, const auto &b) {
    check_old_index_oracle(a, b, tf::cpp::sub(a, b),
                           [](auto x, auto y) { return x - y; });
    check_old_index_oracle(b, a, tf::cpp::sub(b, a),
                           [](auto x, auto y) { return x - y; });
  };

  const auto rank_four =
      make_generated_array<std::int32_t>({2, 1, 4, 3}, [](std::size_t index) {
        return static_cast<std::int32_t>(index * 7 + 1);
      });
  const auto vec3 = make_array<std::int32_t>({5, -2, 11}, {3});
  check_both_orders(rank_four, vec3);

  const auto compact =
      make_generated_array<std::int32_t>({2, 3}, [](std::size_t index) {
        return static_cast<std::int32_t>(index + 20);
      });
  const auto rank_padded =
      make_generated_array<std::int32_t>({1, 1, 2, 3}, [](std::size_t index) {
        return static_cast<std::int32_t>(index * 3 - 4);
      });
  check_both_orders(compact, rank_padded);

  const auto period_two =
      make_generated_array<std::int32_t>({7, 2}, [](std::size_t index) {
        return static_cast<std::int32_t>(index * 5);
      });
  const auto vec2 = make_array<std::int32_t>({4, 9}, {2});
  check_both_orders(period_two, vec2);

  const auto period_four =
      make_generated_array<std::int32_t>({5, 4}, [](std::size_t index) {
        return static_cast<std::int32_t>(index * 2 + 3);
      });
  const auto vec4 = make_array<std::int32_t>({1, 4, 7, 10}, {4});
  check_both_orders(period_four, vec4);

  const auto scalar = make_array<std::int32_t>({13}, {1});
  check_both_orders(period_four, scalar);
}

TEST_CASE("broadcast arithmetic handles scalar singleton and multiple axes",
          "[cpp][core][elementwise][broadcast]") {
  const auto matrix = make_array<std::int32_t>({1, 2, 3, 4, 5, 6}, {2, 3});
  const auto scalar = make_array<std::int32_t>({10}, {1});
  const auto singleton = make_array<std::int32_t>({2, 3}, {2, 1});
  check_values(tf::cpp::sub(matrix, scalar), {-9, -8, -7, -6, -5, -4});
  check_values(tf::cpp::sub(scalar, matrix), {9, 8, 7, 6, 5, 4});
  check_values(tf::cpp::mul(matrix, singleton), {2, 4, 6, 12, 15, 18});

  const auto left =
      make_generated_array<std::int32_t>({2, 1, 3}, [](std::size_t index) {
        return static_cast<std::int32_t>(index + 1);
      });
  const auto right =
      make_generated_array<std::int32_t>({1, 4, 1}, [](std::size_t index) {
        return static_cast<std::int32_t>((index + 1) * 10);
      });
  const auto result = tf::cpp::add(left, right);
  REQUIRE((result.raw_shape() == tf::small_vector<int, 3>{2, 4, 3}));
  for (int outer = 0; outer < 2; ++outer) {
    for (int middle = 0; middle < 4; ++middle) {
      for (int inner = 0; inner < 3; ++inner) {
        const auto index = (outer * 4 + middle) * 3 + inner;
        CHECK(result[static_cast<std::size_t>(index)] ==
              (outer * 3 + inner + 1) + (middle + 1) * 10);
      }
    }
  }
}

TEST_CASE("broadcast comparisons logicals and assignment share traversal",
          "[cpp][core][elementwise][broadcast]") {
  const auto values = make_array<std::int32_t>({1, 5, 3, 4, 2, 8}, {2, 3});
  const auto limits = make_array<std::int32_t>({2, 2, 7}, {3});
  check_values(tf::cpp::gt(values, limits),
               {std::int8_t{0}, std::int8_t{1}, std::int8_t{0}, std::int8_t{1},
                std::int8_t{0}, std::int8_t{1}});

  const auto row_flags =
      make_array<std::int8_t>({std::int8_t{1}, std::int8_t{0}}, {2, 1});
  const auto column_flags = make_array<std::int8_t>(
      {std::int8_t{0}, std::int8_t{1}, std::int8_t{1}}, {1, 3});
  check_values(tf::cpp::logical_and(row_flags, column_flags),
               {std::int8_t{0}, std::int8_t{1}, std::int8_t{1}, std::int8_t{0},
                std::int8_t{0}, std::int8_t{0}});
  check_values(tf::cpp::logical_or(row_flags, column_flags),
               {std::int8_t{1}, std::int8_t{1}, std::int8_t{1}, std::int8_t{0},
                std::int8_t{1}, std::int8_t{1}});

  auto target = make_generated_array<std::int32_t>(
      {2, 4, 3}, [](std::size_t) { return 0; });
  const auto source = make_array<std::int32_t>({10, 20, 30, 40}, {1, 4, 1});
  tf::cpp::assign_array(target, source);
  for (int outer = 0; outer < 2; ++outer)
    for (int middle = 0; middle < 4; ++middle)
      for (int inner = 0; inner < 3; ++inner)
        CHECK(target[static_cast<std::size_t>((outer * 4 + middle) * 3 +
                                              inner)] == (middle + 1) * 10);
}

TEST_CASE("broadcast traversal accepts empty output dimensions",
          "[cpp][core][elementwise][broadcast]") {
  const auto check_empty_both_orders = [](tf::small_vector<int, 3> a_shape,
                                          tf::small_vector<int, 3> b_shape) {
    const auto a =
        make_generated_array<float>(std::move(a_shape), [](std::size_t index) {
          return static_cast<float>(index + 1);
        });
    const auto b =
        make_generated_array<float>(std::move(b_shape), [](std::size_t index) {
          return static_cast<float>(index + 2);
        });
    check_old_index_oracle(a, b, tf::cpp::add(a, b),
                           [](auto x, auto y) { return x + y; });
    check_old_index_oracle(b, a, tf::cpp::add(b, a),
                           [](auto x, auto y) { return x + y; });
  };

  check_empty_both_orders({0, 3}, {3});
  check_empty_both_orders({2, 0, 3}, {1, 1, 3});
  check_empty_both_orders({2, 3, 0}, {1});

  const auto vector = make_array<float>({1.0F, 2.0F, 3.0F}, {3});
  auto assigned = make_array<float>(std::initializer_list<float>{}, {2, 0, 3});
  tf::cpp::assign_array(assigned, vector);
  CHECK((assigned.raw_shape() == tf::small_vector<int, 3>{2, 0, 3}));
  CHECK(assigned.length() == 0);
}

TEST_CASE("broadcast partitions reproduce single-thread output bytes",
          "[cpp][core][elementwise][broadcast]") {
  constexpr int rows = 50001;
  const auto matrix =
      make_generated_array<float>({rows, 3}, [](std::size_t index) {
        return static_cast<float>(static_cast<int>(index % 97) - 48) * 0.25F;
      });
  const auto vector = make_array<float>({0.5F, -1.25F, 2.0F}, {3});

  const auto run = [&](std::size_t thread_cap) {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                thread_cap);
    return tf::cpp::add(matrix, vector);
  };
  const auto single_thread = run(1);
  const auto multi_thread = run(8);
  check_old_index_oracle(matrix, vector, multi_thread,
                         [](auto x, auto y) { return x + y; });
  REQUIRE(single_thread.length() == static_cast<std::size_t>(rows) * 3);
  REQUIRE(multi_thread.length() == single_thread.length());
  CHECK(std::memcmp(single_thread.raw_data(), multi_thread.raw_data(),
                    single_thread.length() * sizeof(float)) == 0);

  auto inplace = matrix.deep_copy();
  {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                8);
    tf::cpp::add_inplace(inplace, vector);
  }
  REQUIRE(inplace.length() == multi_thread.length());
  CHECK(std::memcmp(inplace.raw_data(), multi_thread.raw_data(),
                    inplace.length() * sizeof(float)) == 0);
}

TEST_CASE("generic broadcast partitions initialize correctly inside rows",
          "[cpp][core][elementwise][broadcast]") {
  constexpr int rows = 20003;
  const auto a =
      make_generated_array<std::int32_t>({rows, 1, 3}, [](std::size_t index) {
        return static_cast<std::int32_t>((index * 17 + 5) % 1009);
      });
  const auto b = make_array<std::int32_t>({7, 11, 13, 17}, {1, 4, 1});

  tbb::global_control control(tbb::global_control::max_allowed_parallelism, 8);
  const auto ab = tf::cpp::sub(a, b);
  const auto ba = tf::cpp::sub(b, a);
  check_old_index_oracle(a, b, ab, [](auto x, auto y) { return x - y; });
  check_old_index_oracle(b, a, ba, [](auto x, auto y) { return x - y; });
}

TEST_CASE("broadcast assignment covers flat tile and generic mappings",
          "[cpp][core][elementwise][broadcast]") {
  const auto flat_source =
      make_generated_array<std::int32_t>({2, 3}, [](std::size_t index) {
        return static_cast<std::int32_t>(index * 3 + 1);
      });
  auto rank_padded_target = make_generated_array<std::int32_t>(
      {1, 1, 2, 3}, [](std::size_t) { return 0; });
  tf::cpp::assign_array(rank_padded_target, flat_source);
  check_assignment_old_index_oracle(rank_padded_target, flat_source);

  const auto tile_source = make_array<std::int32_t>({3, 5, 7}, {3});
  auto tile_target = make_generated_array<std::int32_t>(
      {4, 2, 3}, [](std::size_t) { return 0; });
  tf::cpp::assign_array(tile_target, tile_source);
  check_assignment_old_index_oracle(tile_target, tile_source);

  const auto generic_source = make_array<std::int32_t>({10, 20}, {1, 2, 1});
  auto generic_target = make_generated_array<std::int32_t>(
      {3, 2, 3}, [](std::size_t) { return 0; });
  tf::cpp::assign_array(generic_target, generic_source);
  check_assignment_old_index_oracle(generic_target, generic_source);

  auto selected_target = make_generated_array<std::int32_t>(
      {4, 2, 3}, [](std::size_t) { return -1; });
  const auto selected_rows = make_array<std::int32_t>({3, 1}, {2});
  tf::cpp::assign_indexed_array(selected_target, selected_rows, tile_source);
  for (const auto row : {1, 3})
    for (auto column = 0; column < 6; ++column)
      CHECK(selected_target[static_cast<std::size_t>(row * 6 + column)] ==
            tile_source[static_cast<std::size_t>(column % 3)]);

  auto masked_target = make_generated_array<std::int32_t>(
      {4, 2, 3}, [](std::size_t) { return -1; });
  const auto mask = make_array<std::int8_t>(
      {std::int8_t{1}, std::int8_t{0}, std::int8_t{1}, std::int8_t{0}}, {4});
  tf::cpp::assign_masked_array(masked_target, mask, generic_source);
  for (const auto row : {0, 2})
    for (auto middle = 0; middle < 2; ++middle)
      for (auto inner = 0; inner < 3; ++inner)
        CHECK(masked_target[static_cast<std::size_t>(row * 6 + middle * 3 +
                                                     inner)] ==
              generic_source[static_cast<std::size_t>(middle)]);
  for (const auto row : {1, 3})
    for (auto column = 0; column < 6; ++column)
      CHECK(masked_target[static_cast<std::size_t>(row * 6 + column)] == -1);
}

TEST_CASE("arithmetic and scalar forms preserve every storage dtype",
          "[cpp][core][elementwise]") {
  check_arithmetic<std::int8_t>();
  check_arithmetic<std::int32_t>();
  check_arithmetic<std::int64_t>();
  check_arithmetic<float>();
  check_arithmetic<double>();

  const auto wide = std::int64_t{1} << 40;
  const auto wide_values = make_array<std::int64_t>({wide, wide}, {2});
  check_values(tf::cpp::add(wide_values, wide_values), {2 * wide, 2 * wide});
  check_values(tf::cpp::mul_scalar(wide_values, std::int64_t{1024}),
               {wide << 10, wide << 10});
}

TEST_CASE("assignment variants preserve every storage dtype",
          "[cpp][core][elementwise]") {
  check_assignments<std::int8_t>();
  check_assignments<std::int32_t>();
  check_assignments<std::int64_t>();
  check_assignments<float>();
  check_assignments<double>();
}

TEST_CASE("indexed assignment normalizes and validates every row index",
          "[cpp][core][elementwise]") {
  auto target = make_array<float>({0, 0, 0, 0, 0, 0}, {3, 2});
  const auto last = make_array<std::int32_t>({-1}, {1});
  tf::cpp::assign_indexed_scalar(target, last, 9.0F);
  check_values(target, {0.0F, 0.0F, 0.0F, 0.0F, 9.0F, 9.0F});

  const auto middle = make_array<std::int32_t>({-2}, {1});
  const auto values = make_array<float>({4, 5}, {1, 2});
  tf::cpp::assign_indexed_array(target, middle, values);
  check_values(target, {0.0F, 0.0F, 4.0F, 5.0F, 9.0F, 9.0F});

  auto unchanged = make_array<float>({1, 2, 3, 4, 5, 6}, {3, 2});
  const auto partly_invalid = make_array<std::int32_t>({0, 3}, {2});
  CHECK_THROWS_AS(
      tf::cpp::assign_indexed_scalar(unchanged, partly_invalid, 0.0F),
      std::out_of_range);
  check_values(unchanged, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});

  const auto too_negative = make_array<std::int32_t>({-4}, {1});
  CHECK_THROWS_AS(
      tf::cpp::assign_indexed_array(unchanged, too_negative, values),
      std::out_of_range);
}

TEST_CASE("masked assignment requires a 1D row mask",
          "[cpp][core][elementwise]") {
  auto target = make_array<double>({1, 2, 3, 4, 5, 6}, {3, 2});
  const auto matrix_mask = make_array<std::int8_t>(
      {std::int8_t{1}, std::int8_t{0}, std::int8_t{1}}, {3, 1});
  CHECK_THROWS_AS(tf::cpp::assign_masked_scalar(target, matrix_mask, 0.0),
                  std::invalid_argument);

  const auto short_mask =
      make_array<std::int8_t>({std::int8_t{1}, std::int8_t{0}}, {2});
  const auto values = make_array<double>({7, 8}, {2});
  CHECK_THROWS_AS(tf::cpp::assign_masked_array(target, short_mask, values),
                  std::invalid_argument);
  check_values(target, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
}

TEST_CASE("comparisons and logical operations return int8 arrays",
          "[cpp][core][elementwise]") {
  check_comparisons<std::int8_t>();
  check_comparisons<std::int32_t>();
  check_comparisons<std::int64_t>();
  check_comparisons<float>();
  check_comparisons<double>();

  const auto low = std::int64_t{1};
  const auto high = (std::int64_t{1} << 32) + 1;
  const auto wide_high = make_array<std::int64_t>({high}, {1});
  const auto wide_low = make_array<std::int64_t>({low}, {1});
  check_values(tf::cpp::eq(wide_high, wide_low), {std::int8_t{0}});
  check_values(tf::cpp::gt(wide_high, wide_low), {std::int8_t{1}});
  check_values(tf::cpp::lt_scalar(wide_low, high), {std::int8_t{1}});

  const auto a = make_array<std::int8_t>(
      {std::int8_t{0}, std::int8_t{1}, std::int8_t{2}}, {3});
  const auto b = make_array<std::int8_t>(
      {std::int8_t{0}, std::int8_t{0}, std::int8_t{1}}, {3});
  check_values(tf::cpp::logical_not(a),
               {std::int8_t{1}, std::int8_t{0}, std::int8_t{0}});
  check_values(tf::cpp::logical_and(a, b),
               {std::int8_t{0}, std::int8_t{0}, std::int8_t{1}});
  check_values(tf::cpp::logical_or(a, b),
               {std::int8_t{0}, std::int8_t{1}, std::int8_t{1}});
  auto inplace = a.deep_copy();
  tf::cpp::logical_not_inplace(inplace);
  check_values(inplace, {std::int8_t{1}, std::int8_t{0}, std::int8_t{0}});
}

TEST_CASE("float unary math and inplace variants share results",
          "[cpp][core][elementwise]") {
  check_float_unaries<float>();
  check_float_unaries<double>();

  const auto input = make_array<double>({2.0, 3.0}, {2});
  check_values(tf::cpp::pow(input, 3.0), {8.0, 27.0});
  auto inplace = input.deep_copy();
  tf::cpp::pow_inplace(inplace, 2.0);
  check_values(inplace, {4.0, 9.0});

  const auto y = make_array<double>({0.0, 1.0}, {2});
  const auto x = make_array<double>({1.0, 1.0}, {2});
  const auto angles = tf::cpp::atan2(y, x);
  CHECK(angles[0] == Catch::Approx(0.0));
  CHECK(angles[1] == Catch::Approx(std::atan2(1.0, 1.0)));
}

TEST_CASE("general unary and vector operations retain their dtype matrices",
          "[cpp][core][elementwise]") {
  check_general_unaries<std::int8_t>();
  check_general_unaries<std::int32_t>();
  check_general_unaries<std::int64_t>();
  check_general_unaries<float>();
  check_general_unaries<double>();

  check_vector_operations<std::int32_t>();
  check_vector_operations<std::int64_t>();
  check_vector_operations<float>();
  check_vector_operations<double>();

  const auto root = std::int64_t{1} << 21;
  const auto square = root * root;
  const auto wide_row = make_array<std::int64_t>({root, root, root}, {3});
  check_values(tf::cpp::dot(wide_row, wide_row), {3 * square});
  const auto wide_matrix =
      make_array<std::int64_t>({root, root, root, root}, {2, 2});
  check_values(tf::cpp::mat_mul(wide_matrix, wide_matrix),
               {2 * square, 2 * square, 2 * square, 2 * square});
}

TEST_CASE("above-threshold single carriers remain valid for vector kernels",
          "[cpp][core][elementwise][parallel]") {
  const auto inner =
      static_cast<int>(tf::cpp::parallel_threshold + std::size_t{1});
  REQUIRE(static_cast<std::size_t>(inner) > tf::cpp::parallel_threshold);

  const auto a = make_generated_array<float>({1, inner}, [](std::size_t index) {
    return static_cast<float>(static_cast<int>(index % 31) - 15) * 0.125F;
  });
  const auto b = make_generated_array<float>({1, inner}, [](std::size_t index) {
    return static_cast<float>(static_cast<int>(index % 29) - 14) * 0.0625F;
  });
  auto expected_dot = 0.0F;
  for (auto index = 0; index < inner; ++index)
    expected_dot +=
        a[static_cast<std::size_t>(index)] * b[static_cast<std::size_t>(index)];
  const auto dot = run_with_thread_cap(8, [&] { return tf::cpp::dot(a, b); });
  REQUIRE((dot.raw_shape() == tf::small_vector<int, 3>{1}));
  CHECK(dot[0] == expected_dot);

  const auto normalizable =
      make_generated_array<float>({1, inner}, [](std::size_t index) {
        return 0.5F + static_cast<float>(index % 23) * 0.03125F;
      });
  auto squared_norm = 0.0F;
  for (auto index = 0; index < inner; ++index) {
    const auto value = normalizable[static_cast<std::size_t>(index)];
    squared_norm += value * value;
  }
  const auto normalized = run_with_thread_cap(
      8, [&] { return tf::cpp::normalize(normalizable, 1); });
  REQUIRE((normalized.raw_shape() == tf::small_vector<int, 3>{1, inner}));
  const auto norm = std::sqrt(squared_norm);
  for (const auto index : {0, inner / 2, inner - 1})
    CHECK(normalized[static_cast<std::size_t>(index)] ==
          normalizable[static_cast<std::size_t>(index)] / norm);

  const auto matrix_b =
      make_generated_array<float>({inner, 2}, [](std::size_t index) {
        return static_cast<float>(static_cast<int>(index % 19) - 9) * 0.0625F;
      });
  auto expected_column_0 = 0.0F;
  auto expected_column_1 = 0.0F;
  for (auto index = 0; index < inner; ++index) {
    const auto a_value = a[static_cast<std::size_t>(index)];
    expected_column_0 +=
        a_value * matrix_b[static_cast<std::size_t>(index) * 2];
    expected_column_1 +=
        a_value * matrix_b[static_cast<std::size_t>(index) * 2 + 1];
  }
  const auto product =
      run_with_thread_cap(8, [&] { return tf::cpp::mat_mul(a, matrix_b); });
  REQUIRE((product.raw_shape() == tf::small_vector<int, 3>{1, 2}));
  check_values(product, {expected_column_0, expected_column_1});
}

TEST_CASE("work-aware dot partitions batches without splitting accumulation",
          "[cpp][core][elementwise][dot][parallel]") {
  constexpr int batch = 24;
  constexpr int inner = 8192;
  const auto a = make_generated_array<float>({batch, inner}, [](std::size_t i) {
    return static_cast<float>(static_cast<int>((i * 17 + 3) % 257) - 128) *
           0.0625F;
  });
  const auto b = make_generated_array<float>({batch, inner}, [](std::size_t i) {
    return static_cast<float>(static_cast<int>((i * 29 + 11) % 251) - 125) *
           0.03125F;
  });

  const auto single =
      run_with_thread_cap(1, [&] { return tf::cpp::dot(a, b); });
  const auto multi = run_with_thread_cap(8, [&] { return tf::cpp::dot(a, b); });
  check_same_storage(single, multi);
  REQUIRE((multi.raw_shape() == tf::small_vector<int, 3>{batch}));

  constexpr int representative_batch = 7;
  auto expected = 0.0F;
  for (auto inner_index = 0; inner_index < inner; ++inner_index) {
    const auto index =
        static_cast<std::size_t>(representative_batch * inner + inner_index);
    expected += a[index] * b[index];
  }
  CHECK(multi[representative_batch] == expected);
}

TEST_CASE(
    "work-aware axis normalization partitions independent strided vectors",
    "[cpp][core][elementwise][normalize][parallel]") {
  constexpr int outer = 3;
  constexpr int axis_size = 2048;
  constexpr int inner = 32;
  const auto input = make_generated_array<float>(
      {outer, axis_size, inner}, [](std::size_t index) {
        return 0.5F + static_cast<float>((index * 23 + 5) % 127) * 0.0078125F;
      });

  const auto single =
      run_with_thread_cap(1, [&] { return tf::cpp::normalize(input, 1); });
  const auto multi =
      run_with_thread_cap(8, [&] { return tf::cpp::normalize(input, 1); });
  check_same_storage(single, multi);
  REQUIRE(
      (multi.raw_shape() == tf::small_vector<int, 3>{outer, axis_size, inner}));

  constexpr int representative_outer = 1;
  constexpr int representative_inner = 9;
  auto squared_norm = 0.0F;
  for (auto axis_index = 0; axis_index < axis_size; ++axis_index) {
    const auto index = static_cast<std::size_t>(
        (representative_outer * axis_size + axis_index) * inner +
        representative_inner);
    squared_norm += input[index] * input[index];
  }
  const auto expected_norm = std::sqrt(squared_norm);
  for (const auto axis_index : {0, axis_size / 2, axis_size - 1}) {
    const auto index = static_cast<std::size_t>(
        (representative_outer * axis_size + axis_index) * inner +
        representative_inner);
    CHECK(multi[index] == input[index] / expected_norm);
  }
}

TEST_CASE("work-aware mat_mul partitions batch-row carriers in output order",
          "[cpp][core][elementwise][mat_mul][parallel]") {
  constexpr int batch = 3;
  constexpr int rows = 4;
  constexpr int shared = 2048;
  constexpr int columns = 8;
  const auto a =
      make_generated_array<float>({batch, rows, shared}, [](std::size_t index) {
        return static_cast<float>(static_cast<int>((index * 13 + 7) % 97) -
                                  48) *
               0.03125F;
      });
  const auto b = make_generated_array<float>(
      {batch, shared, columns}, [](std::size_t index) {
        return static_cast<float>(static_cast<int>((index * 19 + 3) % 89) -
                                  44) *
               0.015625F;
      });

  const auto single =
      run_with_thread_cap(1, [&] { return tf::cpp::mat_mul(a, b); });
  const auto multi =
      run_with_thread_cap(8, [&] { return tf::cpp::mat_mul(a, b); });
  check_same_storage(single, multi);
  REQUIRE(
      (multi.raw_shape() == tf::small_vector<int, 3>{batch, rows, columns}));

  constexpr int representative_batch = 1;
  constexpr int representative_row = 2;
  constexpr int representative_column = 5;
  auto expected = 0.0F;
  for (auto inner_index = 0; inner_index < shared; ++inner_index) {
    const auto a_index = static_cast<std::size_t>(
        (representative_batch * rows + representative_row) * shared +
        inner_index);
    const auto b_index = static_cast<std::size_t>(
        (representative_batch * shared + inner_index) * columns +
        representative_column);
    expected += a[a_index] * b[b_index];
  }
  const auto output_index = static_cast<std::size_t>(
      (representative_batch * rows + representative_row) * columns +
      representative_column);
  CHECK(multi[output_index] == expected);
}

TEST_CASE("mat_mul broadcasts singleton 3D batches in both directions",
          "[cpp][core][elementwise]") {
  check_singleton_batch_mat_mul<float>();
  check_singleton_batch_mat_mul<double>();
}

TEST_CASE("normalize supports whole arrays and selected axes",
          "[cpp][core][elementwise]") {
  const auto input = make_array<double>({3.0, 4.0, 0.0, 5.0}, {2, 2});
  const auto global = tf::cpp::normalize(input);
  CHECK(global[0] == Catch::Approx(3.0 / std::sqrt(50.0)));
  CHECK(global[3] == Catch::Approx(5.0 / std::sqrt(50.0)));

  const auto rows = tf::cpp::normalize(input, 1);
  CHECK(rows[0] == Catch::Approx(0.6));
  CHECK(rows[1] == Catch::Approx(0.8));
  CHECK(rows[2] == Catch::Approx(0.0));
  CHECK(rows[3] == Catch::Approx(1.0));

  auto inplace = make_array<float>({3.0F, 4.0F}, {2});
  tf::cpp::normalize_inplace(inplace);
  CHECK(inplace[0] == Catch::Approx(0.6F));
  CHECK(inplace[1] == Catch::Approx(0.8F));
  tf::cpp::normalize_inplace(inplace, 0);
  CHECK(inplace[0] == Catch::Approx(0.6F));
  CHECK(inplace[1] == Catch::Approx(0.8F));

  // one producer answers what an axis is, so normalize refuses exactly as
  // take does: out of range for an axis the shape does not have, and no axis
  // at all before that
  CHECK_THROWS_AS(tf::cpp::normalize(input, 2), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::normalize(input, -3), std::out_of_range);
  auto axisless = tf::cpp::nd_array<float>{};
  CHECK_THROWS_AS(tf::cpp::normalize_inplace(axisless, 0),
                  std::invalid_argument);
}

TEST_CASE("all twenty cross-dtype casts link from the native archive",
          "[cpp][core][elementwise]") {
  check_cast<std::int8_t, std::int32_t>();
  check_cast<std::int8_t, std::int64_t>();
  check_cast<std::int8_t, float>();
  check_cast<std::int8_t, double>();
  check_cast<std::int32_t, std::int8_t>();
  check_cast<std::int32_t, std::int64_t>();
  check_cast<std::int32_t, float>();
  check_cast<std::int32_t, double>();
  check_cast<std::int64_t, std::int8_t>();
  check_cast<std::int64_t, std::int32_t>();
  check_cast<std::int64_t, float>();
  check_cast<std::int64_t, double>();
  check_cast<float, std::int8_t>();
  check_cast<float, std::int32_t>();
  check_cast<float, std::int64_t>();
  check_cast<float, double>();
  check_cast<double, std::int8_t>();
  check_cast<double, std::int32_t>();
  check_cast<double, std::int64_t>();
  check_cast<double, float>();

  const auto exact = std::int64_t{1} << 53;
  const auto wide = make_array<std::int64_t>({exact}, {1});
  check_values(tf::cpp::cast<std::int64_t, double>(wide),
               {static_cast<double>(exact)});
  check_values(tf::cpp::cast<double, std::int64_t>(
                   make_array<double>({static_cast<double>(exact)}, {1})),
               {exact});
}

TEST_CASE("async elementwise fronts match sync and resolver-selected results",
          "[cpp][core][async][elementwise]") {
  const auto a = make_array<float>({1, 2, 3, 4}, {2, 2});
  const auto b = make_array<float>({10, 20}, {2});
  const auto expected = tf::cpp::add(a, b);
  check_values(tf::cpp::async::add(a, b).get(), {11.0F, 22.0F, 13.0F, 24.0F});
  check_values(tf::cpp::async::mul_scalar(a, 2.0F).get(),
               {2.0F, 4.0F, 6.0F, 8.0F});
  check_values(
      tf::cpp::async::gte_scalar(a, 3.0F).get(),
      {std::int8_t{0}, std::int8_t{0}, std::int8_t{1}, std::int8_t{1}});

  auto pending = tf::cpp::async::add(tf::cpp::async::future_resolver{}, a, b);
  static_assert(std::is_same_v<decltype(pending), std::future<float_array>>);
  const auto custom = pending.get();
  REQUIRE(custom.length() == expected.length());
  for (std::size_t index = 0; index < custom.length(); ++index)
    CHECK(custom[index] == expected[index]);
}

TEST_CASE("async elementwise in-place work resolves void and reports errors",
          "[cpp][core][async][elementwise]") {
  auto values = make_array<double>({-2.0, 0.5, 3.0}, {3});
  auto clipped = tf::cpp::async::clip_inplace(values, -1.0, 1.0);
  CHECK_NOTHROW(clipped.get());
  check_values(values, {-1.0, 0.5, 1.0});

  const auto a = make_array<std::int32_t>({1, 2}, {2});
  const auto b = make_array<std::int32_t>({1, 2, 3}, {3});
  auto failure = tf::cpp::async::add(a, b);
  CHECK_THROWS_AS(failure.get(), std::runtime_error);
}

TEST_CASE("async elementwise work retains dispatched input storage",
          "[cpp][core][async][elementwise]") {
  auto input = make_array<double>({1.0, 4.0, 9.0}, {3});
  auto result = [&input] {
    tbb::global_control block_workers(
        tbb::global_control::max_allowed_parallelism, 1);
    auto pending = tf::cpp::async::sqrt(input);
    input.destroy();
    return pending;
  }();
  check_values(result.get(), {1.0, 2.0, 3.0});
}

TEST_CASE("affine inversion exposes sync future and custom resolver facades",
          "[cpp][core][async][elementwise][inverted]") {
  check_inverted_facades<float>();
  check_inverted_facades<double>();
}

TEST_CASE("affine inversion validates arrays without a singularity policy",
          "[cpp][core][elementwise][inverted]") {
  const tf::cpp::nd_array<float> invalid;
  CHECK_THROWS_AS(tf::cpp::inverted(invalid), std::invalid_argument);

  const auto flat = make_array<float>({1, 0, 0, 0, 1, 0, 0, 0, 1}, {9});
  const auto nonsquare =
      make_array<double>({1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {3, 4});
  const auto wrong_size = make_array<double>({1, 0, 0, 1}, {2, 2});
  CHECK_THROWS_AS(tf::cpp::inverted(flat), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::inverted(nonsquare), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::inverted(wrong_size), std::invalid_argument);

  auto async_invalid = tf::cpp::async::inverted(invalid);
  CHECK_THROWS_AS(async_invalid.get(), std::invalid_argument);
  auto async_wrong_size = tf::cpp::async::inverted(wrong_size);
  CHECK_THROWS_AS(async_wrong_size.get(), std::invalid_argument);

  const auto singular_3x3 =
      make_array<float>({1, 0, 2, 0, 0, 3, 9, 8, 7}, {3, 3});
  const auto singular_4x4 = make_array<double>(
      {1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 0, 4, 9, 8, 7, 6}, {4, 4});
  tf::cpp::nd_array<float> result_3x3;
  tf::cpp::nd_array<double> result_4x4;
  CHECK_NOTHROW(result_3x3 = tf::cpp::inverted(singular_3x3));
  CHECK_NOTHROW(result_4x4 = tf::cpp::inverted(singular_4x4));
  CHECK(result_3x3[6] == 0.0F);
  CHECK(result_3x3[7] == 0.0F);
  CHECK(result_3x3[8] == 1.0F);
  CHECK(result_4x4[12] == 0.0);
  CHECK(result_4x4[13] == 0.0);
  CHECK(result_4x4[14] == 0.0);
  CHECK(result_4x4[15] == 1.0);
}

TEST_CASE("async affine inversion retains input before one custom submission",
          "[cpp][core][async][elementwise][inverted]") {
  auto submissions = 0;

  auto matrix_3x3 = make_array<float>({2, 0, 4, 0, 4, 8, 0, 0, 1}, {3, 3});
  auto pending_3x3 = tf::cpp::async::inverted(
      destroying_resolver<float>{&submissions, &matrix_3x3}, matrix_3x3);
  CHECK(submissions == 1);
  CHECK_FALSE(matrix_3x3.is_valid());
  check_inverted_result(
      pending_3x3.get(), 3,
      {0.5F, 0.0F, -2.0F, 0.0F, 0.25F, -2.0F, 0.0F, 0.0F, 1.0F});

  auto matrix_4x4 = make_array<double>(
      {2, 0, 0, 6, 0, 4, 0, 8, 0, 0, 5, 10, 0, 0, 0, 1}, {4, 4});
  auto pending_4x4 = tf::cpp::async::inverted(
      destroying_resolver<double>{&submissions, &matrix_4x4}, matrix_4x4);
  CHECK(submissions == 2);
  CHECK_FALSE(matrix_4x4.is_valid());
  check_inverted_result(pending_4x4.get(), 4,
                        {0.5, 0.0, 0.0, -3.0, 0.0, 0.25, 0.0, -2.0, 0.0, 0.0,
                         0.2, -2.0, 0.0, 0.0, 0.0, 1.0});
}
