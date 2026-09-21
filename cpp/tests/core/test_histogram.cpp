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
#include "trueform/cpp/core/async/histogram.hpp"
#include "trueform/cpp/core/histogram.hpp"
#include "trueform/cpp/core/parallel_config.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <tbb/global_control.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <initializer_list>
#include <limits>
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

template <typename T>
auto check_values(const tf::cpp::nd_array<T> &array,
                  std::initializer_list<T> expected) -> void {
  REQUIRE(array.length() == expected.size());
  auto index = std::size_t{0};
  for (const auto value : expected)
    CHECK(array[index++] == value);
}

template <typename T, typename Generator>
auto make_generated_array(std::size_t length, Generator generator)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(length);
  for (std::size_t index = 0; index < length; ++index)
    buffer[index] = generator(index);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer),
                                           {static_cast<int>(length)});
}

template <typename T>
auto check_same_storage(const tf::cpp::nd_array<T> &a,
                        const tf::cpp::nd_array<T> &b) -> void {
  REQUIRE(a.raw_shape() == b.raw_shape());
  REQUIRE(a.length() == b.length());
  CHECK(std::memcmp(a.raw_data(), b.raw_data(), a.length() * sizeof(T)) == 0);
}

template <typename T>
auto check_reference_storage(const tf::cpp::nd_array<T> &actual,
                             const std::vector<T> &expected) -> void {
  REQUIRE(actual.length() == expected.size());
  CHECK(std::memcmp(actual.raw_data(), expected.data(),
                    expected.size() * sizeof(T)) == 0);
}

template <typename Operation>
auto run_with_thread_cap(std::size_t thread_cap, Operation operation)
    -> decltype(operation()) {
  tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                              thread_cap);
  return operation();
}

auto check_count_oracle(const tf::cpp::nd_array<std::int32_t> &actual,
                        const std::vector<std::int32_t> &expected,
                        std::int32_t expected_total) -> void {
  check_reference_storage(actual, expected);
  auto total = std::int32_t{0};
  for (const auto count : actual)
    total += count;
  CHECK(total == expected_total);
}

template <typename BinFunction>
auto weighted_reference(const tf::cpp::nd_array<float> &values,
                        const tf::cpp::nd_array<float> &weights, int bin_count,
                        BinFunction bin_function) -> std::vector<float> {
  std::vector<float> counts(static_cast<std::size_t>(bin_count), 0.0F);
  const auto *value_data = values.raw_data();
  const auto *weight_data = weights.raw_data();
  for (std::size_t index = 0; index < values.length(); ++index) {
    const auto bin = bin_function(value_data[index]);
    if (bin >= 0)
      counts[static_cast<std::size_t>(bin)] += weight_data[index];
  }
  return counts;
}

auto density_reference(std::vector<float> counts,
                       const std::vector<float> &edges) -> std::vector<float> {
  double total = 0.0;
  for (const auto count : counts)
    total += static_cast<double>(count);
  if (total <= 0.0)
    return counts;
  for (std::size_t index = 0; index < counts.size(); ++index) {
    const auto width = static_cast<double>(edges[index + 1]) -
                       static_cast<double>(edges[index]);
    counts[index] =
        width > 0.0 ? static_cast<float>(static_cast<double>(counts[index]) /
                                         (total * width))
                    : 0.0F;
  }
  return counts;
}

template <typename T> auto check_bincount_dtype() -> void {
  const auto values = make_array<T>({1, 1, 2, 3, 3, 3}, {6});
  const auto counts = tf::cpp::bincount(values);
  CHECK((counts.raw_shape() == tf::small_vector<int, 3>{4}));
  check_values(counts, {T{0}, T{2}, T{1}, T{3}});

  const auto weights = make_array<float>({0.5F, 1, 2, 3, 4, 5}, {6});
  const auto weighted = tf::cpp::bincount(values, weights, 6);
  CHECK((weighted.raw_shape() == tf::small_vector<int, 3>{6}));
  check_values(weighted, {0.0F, 1.5F, 2.0F, 12.0F, 0.0F, 0.0F});

  const auto empty = make_array<T>({}, {0});
  check_values(tf::cpp::bincount(empty, 3), {T{0}, T{0}, T{0}});
  const auto singleton = make_array<T>({4}, {1});
  check_values(tf::cpp::bincount(singleton), {T{0}, T{0}, T{0}, T{0}, T{1}});
}

template <typename T> auto check_bincount_refusals() -> void {
  const auto negative = make_array<T>({0, -1}, {2});
  CHECK_THROWS_AS(tf::cpp::bincount(negative), std::invalid_argument);
  const auto values = make_array<T>({0, 1}, {2});
  const auto short_weights = make_array<float>({1}, {1});
  CHECK_THROWS_AS(tf::cpp::bincount(values, -1), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::bincount(values, short_weights),
                  std::invalid_argument);
}

auto density_integral(const tf::cpp::histogram_result<float> &result)
    -> double {
  double integral = 0.0;
  for (std::size_t index = 0; index < result.counts.length(); ++index)
    integral +=
        static_cast<double>(result.counts[index]) *
        static_cast<double>(result.edges[index + 1] - result.edges[index]);
  return integral;
}

} // namespace

TEST_CASE("bincount returns native integer and weighted arrays",
          "[cpp][core][histogram]") {
  check_bincount_dtype<std::int32_t>();
  check_bincount_dtype<std::int64_t>();
}

TEST_CASE("bincount validates values, lengths, and minimum length",
          "[cpp][core][histogram]") {
  check_bincount_refusals<std::int32_t>();
  check_bincount_refusals<std::int64_t>();
  const auto too_large =
      make_array<std::int32_t>({std::numeric_limits<std::int32_t>::max()}, {1});
  CHECK_THROWS_AS(tf::cpp::bincount(too_large), std::length_error);
}

TEST_CASE("bincount counts an id into its own bin at the id's own width",
          "[cpp][core][histogram]") {
  const auto beyond = make_array<std::int64_t>({std::int64_t{1} << 40}, {1});
  CHECK_THROWS_AS(tf::cpp::bincount(beyond), std::length_error);

  const auto aliasing =
      make_array<std::int64_t>({3, (std::int64_t{1} << 32) + 5}, {2});
  const auto weights = make_array<float>({1, 1}, {2});
  CHECK_THROWS_AS(tf::cpp::bincount(aliasing), std::length_error);
  CHECK_THROWS_AS(tf::cpp::bincount(aliasing, 6), std::length_error);
  CHECK_THROWS_AS(tf::cpp::bincount(aliasing, weights), std::length_error);

  const auto high = (std::int64_t{1} << 20) + 1;
  const auto ids = make_array<std::int64_t>({high, 3, high}, {3});
  const auto counts = tf::cpp::bincount(ids);
  static_assert(
      std::is_same_v<decltype(counts), const tf::cpp::nd_array<std::int64_t>>);
  REQUIRE(counts.length() == static_cast<std::size_t>(high) + 1);
  CHECK(counts[static_cast<std::size_t>(high)] == 2);
  CHECK(counts[3] == 1);
  CHECK(counts[5] == 0);
}

TEST_CASE("equal-width histograms cover count, weighted, and density results",
          "[cpp][core][histogram]") {
  const auto values =
      make_array<float>({-5.0F, 0.0F, 0.25F, 0.5F, 0.75F, 1.0F,
                         std::numeric_limits<float>::quiet_NaN(), 5.0F},
                        {8});
  const auto result = tf::cpp::histogram_equal_width(values, 4, 0, 1);
  check_values(result.counts, {std::int32_t{1}, std::int32_t{1},
                               std::int32_t{1}, std::int32_t{2}});
  check_values(result.edges, {0.0F, 0.25F, 0.5F, 0.75F, 1.0F});

  const auto weights = make_array<float>({1, 2, 3, 4, 5, 6, 7, 8}, {8});
  const auto weighted =
      tf::cpp::histogram_equal_width(values, weights, 4, 0, 1);
  check_values(weighted.counts, {2.0F, 3.0F, 4.0F, 11.0F});

  const auto density = tf::cpp::histogram_density_equal_width(
      values, 4, 0, 1, weights.shallow_copy());
  CHECK(density_integral(density) == Catch::Approx(1.0));

  const auto automatic_values = make_array<float>({-1, 0, 1, 3}, {4});
  const auto automatic = tf::cpp::histogram_equal_width(
      automatic_values, 2, std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN());
  CHECK(automatic.edges[0] == -1.0F);
  CHECK(automatic.edges[2] == 3.0F);

  const auto constant = make_array<float>({2, 2}, {2});
  const auto expanded = tf::cpp::histogram_equal_width(
      constant, 2, std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN());
  CHECK(expanded.edges[0] == 1.5F);
  CHECK(expanded.edges[2] == 2.5F);
  check_values(expanded.counts, {std::int32_t{0}, std::int32_t{2}});

  const auto empty = make_array<float>({}, {0});
  const auto empty_result = tf::cpp::histogram_equal_width(
      empty, 2, std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN());
  check_values(empty_result.counts, {std::int32_t{0}, std::int32_t{0}});
  check_values(empty_result.edges, {0.0F, 0.5F, 1.0F});

  const auto wide =
      tf::cpp::histogram_equal_width<std::int64_t>(values, 4, 0, 1);
  static_assert(std::is_same_v<decltype(wide),
                               const tf::cpp::histogram_result<std::int64_t>>);
  check_values(wide.counts, {std::int64_t{1}, std::int64_t{1}, std::int64_t{1},
                             std::int64_t{2}});
  check_values(wide.edges, {0.0F, 0.25F, 0.5F, 0.75F, 1.0F});
}

TEST_CASE("explicit-edge histograms preserve edges and unequal density widths",
          "[cpp][core][histogram]") {
  const auto values = make_array<float>({0.5F, 2, 2.5F, 9.9F, 10}, {5});
  auto edges = make_array<float>({0, 1, 3, 10}, {4});
  const auto result = tf::cpp::histogram_edges(values, edges);
  check_values(result.counts,
               {std::int32_t{1}, std::int32_t{2}, std::int32_t{2}});
  CHECK(result.edges.raw_owner() == edges.raw_owner());

  const auto weights = make_array<float>({1, 2, 3, 4, 5}, {5});
  const auto weighted = tf::cpp::histogram_edges(values, weights, edges);
  check_values(weighted.counts, {1.0F, 5.0F, 9.0F});
  const auto density =
      tf::cpp::histogram_density_edges(values, edges, weights.shallow_copy());
  CHECK(density_integral(density) == Catch::Approx(1.0));

  const auto repeated_edges = make_array<float>({0, 1, 1, 2}, {4});
  const auto repeated = tf::cpp::histogram_edges(values, repeated_edges);
  CHECK(repeated.counts.length() == 3);

  const auto wide = tf::cpp::histogram_edges<std::int64_t>(values, edges);
  static_assert(std::is_same_v<decltype(wide),
                               const tf::cpp::histogram_result<std::int64_t>>);
  check_values(wide.counts,
               {std::int64_t{1}, std::int64_t{2}, std::int64_t{2}});
  CHECK(wide.edges.raw_owner() == edges.raw_owner());
}

TEST_CASE("large unweighted histograms are exact across thread caps",
          "[cpp][core][histogram][parallel]") {
  const auto length = tf::cpp::parallel_threshold + std::size_t{12'347};
  const auto values =
      make_generated_array<float>(length, [](std::size_t index) {
        switch (index % 1021) {
        case 0:
          return std::numeric_limits<float>::quiet_NaN();
        case 1:
          return -0.1F;
        case 2:
          return 1.1F;
        case 3:
          return 1.0F;
        case 4:
          return 0.0F;
        case 5:
          return 0.25F;
        default:
          return static_cast<float>((index * 48271ULL + 17ULL) % 100000ULL) /
                 99999.0F;
        }
      });

  SECTION("equal width") {
    constexpr int bin_count = 64;
    std::vector<std::int32_t> oracle(bin_count, 0);
    auto oracle_total = std::int32_t{0};
    for (const auto value : values) {
      if (std::isnan(value) || value < 0.0F || value > 1.0F)
        continue;
      const auto bin =
          value == 1.0F
              ? bin_count - 1
              : std::min(static_cast<int>(value * bin_count), bin_count - 1);
      ++oracle[static_cast<std::size_t>(bin)];
      ++oracle_total;
    }

    const auto cap_one = run_with_thread_cap(1, [&] {
      return tf::cpp::histogram_equal_width(values, bin_count, 0.0F, 1.0F);
    });
    const auto cap_eight = run_with_thread_cap(8, [&] {
      return tf::cpp::histogram_equal_width(values, bin_count, 0.0F, 1.0F);
    });
    check_same_storage(cap_one.counts, cap_eight.counts);
    check_same_storage(cap_one.edges, cap_eight.edges);
    check_count_oracle(cap_one.counts, oracle, oracle_total);
    check_count_oracle(cap_eight.counts, oracle, oracle_total);

    const auto wide = run_with_thread_cap(8, [&] {
      return tf::cpp::histogram_equal_width<std::int64_t>(values, bin_count,
                                                          0.0F, 1.0F);
    });
    REQUIRE(wide.counts.length() == oracle.size());
    for (std::size_t bin = 0; bin < oracle.size(); ++bin)
      CHECK(wide.counts[bin] == static_cast<std::int64_t>(oracle[bin]));
  }

  SECTION("explicit edges") {
    constexpr int bin_count = 32;
    const auto edges =
        make_generated_array<float>(bin_count + 1, [](std::size_t index) {
          const auto ratio = static_cast<float>(index) / bin_count;
          return ratio * ratio;
        });
    std::vector<std::int32_t> oracle(bin_count, 0);
    auto oracle_total = std::int32_t{0};
    for (const auto value : values) {
      if (std::isnan(value) || value < edges[0] || value > edges[bin_count])
        continue;
      for (int bin = 0; bin < bin_count; ++bin) {
        if (value < edges[bin + 1] ||
            (bin == bin_count - 1 && value == edges[bin_count])) {
          ++oracle[static_cast<std::size_t>(bin)];
          ++oracle_total;
          break;
        }
      }
    }

    const auto cap_one = run_with_thread_cap(
        1, [&] { return tf::cpp::histogram_edges(values, edges); });
    const auto cap_eight = run_with_thread_cap(
        8, [&] { return tf::cpp::histogram_edges(values, edges); });
    check_same_storage(cap_one.counts, cap_eight.counts);
    check_same_storage(cap_one.edges, cap_eight.edges);
    check_count_oracle(cap_one.counts, oracle, oracle_total);
    check_count_oracle(cap_eight.counts, oracle, oracle_total);
  }
}

TEST_CASE("weighted histogram accumulation remains in serial input order",
          "[cpp][core][histogram][parallel]") {
  const auto length = tf::cpp::parallel_threshold + std::size_t{37};
  const auto values =
      make_generated_array<float>(length, [](std::size_t index) {
        switch (index % 8) {
        case 0:
        case 1:
        case 2:
        case 3:
          return 0.25F;
        case 4:
        case 5:
          return 0.75F;
        case 6:
          return 1.0F;
        default:
          return index % 16 == 7 ? std::numeric_limits<float>::quiet_NaN()
                                 : -1.0F;
        }
      });
  const auto weights =
      make_generated_array<float>(length, [](std::size_t index) {
        switch (index % 8) {
        case 0:
          return 1.0e20F;
        case 1:
          return 1.0F;
        case 2:
          return -1.0e20F;
        case 3:
          return 3.0F;
        case 4:
          return 1.0e10F;
        case 5:
          return 1.0F;
        case 6:
          return 0.5F;
        default:
          return 7.0e19F;
        }
      });
  const std::vector<float> edge_values{0.0F, 0.5F, 1.0F};
  const auto edges = make_array<float>({0.0F, 0.5F, 1.0F}, {3});
  const auto classify = [](float value) {
    if (std::isnan(value) || value < 0.0F || value > 1.0F)
      return -1;
    if (value < 0.5F)
      return 0;
    return 1;
  };
  const auto reference = weighted_reference(values, weights, 2, classify);
  const auto density = density_reference(reference, edge_values);

  const auto check_equal_width = [&](std::size_t thread_cap) {
    const auto weighted = run_with_thread_cap(thread_cap, [&] {
      return tf::cpp::histogram_equal_width(values, weights, 2, 0.0F, 1.0F);
    });
    check_reference_storage(weighted.counts, reference);
    const auto weighted_density = run_with_thread_cap(thread_cap, [&] {
      return tf::cpp::histogram_density_equal_width(values, 2, 0.0F, 1.0F,
                                                    weights.shallow_copy());
    });
    check_reference_storage(weighted_density.counts, density);
  };
  check_equal_width(1);
  check_equal_width(8);

  const auto check_explicit_edges = [&](std::size_t thread_cap) {
    const auto weighted = run_with_thread_cap(thread_cap, [&] {
      return tf::cpp::histogram_edges(values, weights, edges);
    });
    check_reference_storage(weighted.counts, reference);
    const auto weighted_density = run_with_thread_cap(thread_cap, [&] {
      return tf::cpp::histogram_density_edges(values, edges,
                                              weights.shallow_copy());
    });
    check_reference_storage(weighted_density.counts, density);
  };
  check_explicit_edges(1);
  check_explicit_edges(8);
}

TEST_CASE("large bin domains retain exact serial histogram fallback",
          "[cpp][core][histogram][parallel]") {
  constexpr int bin_count = 16'384;
  const auto length = tf::cpp::parallel_threshold + std::size_t{1};
  const auto values =
      make_generated_array<float>(length, [](std::size_t index) {
        if (index % 4093 == 0)
          return std::numeric_limits<float>::quiet_NaN();
        if (index % 4093 == 1)
          return 1.0F;
        return static_cast<float>((index * 8191ULL) % 1'000'003ULL) /
               1'000'003.0F;
      });
  std::vector<std::int32_t> oracle(bin_count, 0);
  auto oracle_total = std::int32_t{0};
  for (const auto value : values) {
    if (std::isnan(value) || value < 0.0F || value > 1.0F)
      continue;
    const auto bin =
        value == 1.0F
            ? bin_count - 1
            : std::min(static_cast<int>(value * bin_count), bin_count - 1);
    ++oracle[static_cast<std::size_t>(bin)];
    ++oracle_total;
  }

  const auto cap_one = run_with_thread_cap(1, [&] {
    return tf::cpp::histogram_equal_width(values, bin_count, 0.0F, 1.0F);
  });
  const auto cap_eight = run_with_thread_cap(8, [&] {
    return tf::cpp::histogram_equal_width(values, bin_count, 0.0F, 1.0F);
  });
  check_same_storage(cap_one.counts, cap_eight.counts);
  check_count_oracle(cap_one.counts, oracle, oracle_total);
}

TEST_CASE("histograms reject invalid bins, ranges, edges, and weights",
          "[cpp][core][histogram]") {
  const auto values = make_array<float>({0, 1}, {2});
  const auto weights = make_array<float>({1}, {1});
  CHECK_THROWS_AS(tf::cpp::histogram_equal_width(values, 0, 0, 1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::histogram_equal_width(values, 2, 1, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::histogram_equal_width(values, weights, 2, 0, 1),
                  std::invalid_argument);

  const auto short_edges = make_array<float>({0}, {1});
  const auto rank_two_edges = make_array<float>({0, 1}, {1, 2});
  const auto decreasing_edges = make_array<float>({0, 2, 1}, {3});
  const auto nan_edges =
      make_array<float>({0, std::numeric_limits<float>::quiet_NaN(), 1}, {3});
  CHECK_THROWS_AS(tf::cpp::histogram_edges(values, short_edges),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::histogram_edges(values, rank_two_edges),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::histogram_edges(values, decreasing_edges),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::histogram_edges(values, nan_edges),
                  std::invalid_argument);
  const auto valid_edges = make_array<float>({0, 1, 2}, {3});
  CHECK_THROWS_AS(tf::cpp::histogram_edges(values, weights, valid_edges),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::histogram_density_edges(values, valid_edges,
                                                   weights.shallow_copy()),
                  std::invalid_argument);
}

TEST_CASE("async histograms retain inputs and preserve native result types",
          "[cpp][core][async][histogram]") {
  auto values = make_array<std::int32_t>({1, 1, 2, 3}, {4});
  auto counts_future = tf::cpp::async::bincount(values);
  static_assert(std::is_same_v<decltype(counts_future),
                               std::future<tf::cpp::nd_array<std::int32_t>>>);
  values.destroy();
  check_values(counts_future.get(), {std::int32_t{0}, std::int32_t{2},
                                     std::int32_t{1}, std::int32_t{1}});

  auto wide_values = make_array<std::int64_t>({1, 1, 2, 3}, {4});
  auto wide_counts_future = tf::cpp::async::bincount(wide_values);
  static_assert(std::is_same_v<decltype(wide_counts_future),
                               std::future<tf::cpp::nd_array<std::int64_t>>>);
  wide_values.destroy();
  check_values(wide_counts_future.get(), {std::int64_t{0}, std::int64_t{2},
                                          std::int64_t{1}, std::int64_t{1}});

  const auto samples = make_array<float>({0, 0.25F, 0.75F, 1}, {4});
  const auto weights = make_array<float>({1, 2, 3, 4}, {4});
  auto weighted =
      tf::cpp::async::histogram_equal_width(samples, weights, 2, 0, 1);
  static_assert(std::is_same_v<decltype(weighted),
                               std::future<tf::cpp::histogram_result<float>>>);
  check_values(weighted.get().counts, {3.0F, 7.0F});

  const auto edges = make_array<float>({0, 0.5F, 1}, {3});
  auto custom_pending = tf::cpp::async::histogram_edges(
      tf::cpp::async::future_resolver{}, samples, edges);
  static_assert(
      std::is_same_v<decltype(custom_pending),
                     std::future<tf::cpp::histogram_result<std::int32_t>>>);
  const auto custom = custom_pending.get();
  check_values(custom.counts, {std::int32_t{2}, std::int32_t{2}});

  auto wide_pending = tf::cpp::async::histogram_edges<std::int64_t>(
      tf::cpp::async::future_resolver{}, samples, edges);
  static_assert(
      std::is_same_v<decltype(wide_pending),
                     std::future<tf::cpp::histogram_result<std::int64_t>>>);
  check_values(wide_pending.get().counts, {std::int64_t{2}, std::int64_t{2}});

  auto wide_bins =
      tf::cpp::async::histogram_equal_width<std::int64_t>(samples, 2, 0, 1);
  static_assert(
      std::is_same_v<decltype(wide_bins),
                     std::future<tf::cpp::histogram_result<std::int64_t>>>);
  check_values(wide_bins.get().counts, {std::int64_t{2}, std::int64_t{2}});

  auto density = tf::cpp::async::histogram_density_edges(
      samples, edges, weights.shallow_copy());
  CHECK(density_integral(density.get()) == Catch::Approx(1.0));

  auto failure = tf::cpp::async::histogram_equal_width(samples, 0, 0, 1);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}
