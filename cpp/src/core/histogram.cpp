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
#include "trueform/cpp/core/histogram.hpp"

#include "../core/checked_work.hpp"

#include "trueform/core/algorithm/block_reduce.hpp"
#include "trueform/core/algorithm/parallel_contains.hpp"
#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/views/sequence_range.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace {

/// A partial table per block is what makes a count parallel, so a table too
/// large to replicate has none: `checked_work(0)` is the cost that keeps a
/// whole range serial, and everything else lets the primitive choose its own
/// width.
///
/// A WEIGHTED COUNT IS ALWAYS SERIAL. Its accumulator is a float sum, which is
/// not associative, and this tier states its weighted counts in the input's
/// own order — the law its fixture proves with catastrophic weights. An
/// integer count is order-independent and takes the width the primitive picks.
constexpr std::size_t histogram_local_count_budget = 65'536;

template <typename Count>
auto count_schedule(std::size_t bin_count) -> tf::checked_t {
  if constexpr (std::is_floating_point_v<Count>) {
    static_cast<void>(bin_count);
    return checked_work(0);
  } else {
    return bin_count == 0 || bin_count > histogram_local_count_budget
               ? checked_work(0)
               : checked_work(1);
  }
}

auto maximum_nonnegative(const nd_array<std::int32_t> &values) -> int {
  const auto negative = tf::parallel_contains(
      values.make_range(), [](std::int32_t value) { return value < 0; },
      checked_work(1));
  if (negative)
    throw std::invalid_argument("bincount: input contains a negative value");
  int maximum = -1;
  tf::blocked_reduce(
      values.make_range(), maximum, int{-1},
      [](const auto &block, int &local) {
        for (const auto value : block)
          local = std::max(local, static_cast<int>(value));
      },
      [](int local, int &output) { output = std::max(output, local); },
      checked_work(1));
  return maximum;
}

template <typename T>
auto require_weights(const nd_array<T> &values, const nd_array<float> &weights,
                     const char *operation) -> void {
  if (values.length() != weights.length())
    throw std::invalid_argument(std::string(operation) +
                                ": weights must have the same length as x");
}

auto finite_min_max(const nd_array<float> &values) -> std::pair<float, float> {
  const auto empty =
      std::pair<float, float>{std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()};
  auto bounds = empty;
  tf::blocked_reduce(
      values.make_range(), bounds, empty,
      [](const auto &block, std::pair<float, float> &local) {
        for (const auto value : block) {
          if (std::isnan(value))
            continue;
          local.first = std::min(local.first, value);
          local.second = std::max(local.second, value);
        }
      },
      [](const std::pair<float, float> &local, std::pair<float, float> &out) {
        out.first = std::min(out.first, local.first);
        out.second = std::max(out.second, local.second);
      },
      checked_work(1));
  return bounds;
}

auto resolve_range(const nd_array<float> &values, float lower, float upper)
    -> std::pair<float, float> {
  const auto automatic_lower = std::isnan(lower);
  const auto automatic_upper = std::isnan(upper);
  if (automatic_lower || automatic_upper) {
    const auto bounds = finite_min_max(values);
    if (!std::isfinite(bounds.first) || !std::isfinite(bounds.second)) {
      if (automatic_lower)
        lower = 0.0F;
      if (automatic_upper)
        upper = 1.0F;
    } else {
      if (automatic_lower)
        lower = bounds.first;
      if (automatic_upper)
        upper = bounds.second;
    }
  }
  if (upper < lower)
    throw std::invalid_argument(
        "histogram: upper range must not be less than lower range");
  if (upper == lower) {
    lower -= 0.5F;
    upper += 0.5F;
  }
  return {lower, upper};
}

auto linear_edges(float lower, float upper, int bin_count) -> nd_array<float> {
  tf::buffer<float> buffer;
  buffer.allocate(static_cast<std::size_t>(bin_count + 1));
  const auto width = (upper - lower) / static_cast<float>(bin_count);
  for (int index = 0; index <= bin_count; ++index)
    buffer[static_cast<std::size_t>(index)] =
        lower + static_cast<float>(index) * width;
  buffer[static_cast<std::size_t>(bin_count)] = upper;
  return nd_array<float>::from_buffer(std::move(buffer), {bin_count + 1});
}

auto equal_width_bin(float value, float lower, float upper, float inverse_width,
                     int bin_count) -> int {
  if (std::isnan(value) || value < lower || value > upper)
    return -1;
  const auto bin = value == upper
                       ? bin_count - 1
                       : static_cast<int>((value - lower) * inverse_width);
  return bin >= bin_count ? bin_count - 1 : bin;
}

auto require_edges(const nd_array<float> &edges) -> int {
  if (edges.ndim() != 1 || edges.length() < 2)
    throw std::invalid_argument(
        "histogram: edges must be a 1D array with length >= 2");
  const auto *edge_data = edges.raw_data();
  for (std::size_t index = 0; index < edges.length(); ++index) {
    if (std::isnan(edge_data[index]))
      throw std::invalid_argument("histogram: edges must not contain NaN");
    if (index != 0 && edge_data[index] < edge_data[index - 1])
      throw std::invalid_argument("histogram: edges must be monotone");
  }
  return static_cast<int>(edges.length()) - 1;
}

auto explicit_edge_bin(float value, const float *edges, int bin_count) -> int {
  if (std::isnan(value))
    return -1;
  const auto iterator = std::upper_bound(edges, edges + bin_count + 1, value);
  const auto bin = static_cast<int>(iterator - edges) - 1;
  if (bin < 0)
    return -1;
  if (bin < bin_count)
    return bin;
  return value == edges[bin_count] ? bin_count - 1 : -1;
}

/// @brief The one counting kernel: a partial table per block, merged after.
///
/// Weighted and unweighted are one operation and one execution shape; what a
/// weight adds is where the increment comes from.
template <typename Count, typename Value, typename BinFunction>
auto count_binned(const nd_array<Value> &values, const nd_array<float> *weights,
                  int bin_count, BinFunction bin_function)
    -> tf::buffer<Count> {
  const auto count_size = static_cast<std::size_t>(bin_count);
  tf::buffer<Count> counts;
  counts.allocate_and_initialize(count_size, Count{0});
  const auto *value_data = values.raw_data();
  const auto *weight_data = weights ? weights->raw_data() : nullptr;

  tf::buffer<Count> local_counts;
  local_counts.allocate_and_initialize(count_size, Count{0});
  tf::blocked_reduce(
      values.make_range(), counts, std::move(local_counts),
      [=](const auto &block, auto &local) {
        auto *local_data = local.data();
        const auto offset =
            static_cast<std::size_t>(block.begin() - value_data);
        for (std::size_t index = 0; index < block.size(); ++index) {
          const auto bin = bin_function(block[index]);
          if (bin >= 0)
            local_data[static_cast<std::size_t>(bin)] +=
                weight_data ? static_cast<Count>(weight_data[offset + index])
                            : Count{1};
        }
      },
      [](const auto &local, auto &output) {
        const auto *local_data = local.data();
        auto *output_data = output.data();
        for (std::size_t bin = 0; bin < output.size(); ++bin)
          output_data[bin] += local_data[bin];
      },
      count_schedule<Count>(count_size));
  return counts;
}

template <typename Count>
auto count_equal_width(const nd_array<float> &values,
                       const nd_array<float> *weights, int bin_count,
                       float lower, float upper) -> tf::buffer<Count> {
  const auto inverse_width = static_cast<float>(bin_count) / (upper - lower);
  return count_binned<Count>(values, weights, bin_count, [=](float value) {
    return equal_width_bin(value, lower, upper, inverse_width, bin_count);
  });
}

template <typename Count>
auto count_edges(const nd_array<float> &values, const nd_array<float> *weights,
                 const nd_array<float> &edges, int bin_count)
    -> tf::buffer<Count> {
  const auto *edge_data = edges.raw_data();
  return count_binned<Count>(values, weights, bin_count, [=](float value) {
    return explicit_edge_bin(value, edge_data, bin_count);
  });
}

auto apply_density(nd_array<float> &counts, const nd_array<float> &edges)
    -> void {
  // The total is a floating-point sum, so it is taken in the input's own order
  // for the same reason a weighted count is.
  double total = 0.0;
  for (const auto count : counts)
    total += static_cast<double>(count);
  if (total <= 0.0)
    return;
  auto *values = counts.raw_data();
  const auto *edge_data = edges.raw_data();
  tf::parallel_for_each(
      tf::make_sequence_range(static_cast<int>(counts.length())),
      [=](int index) {
        const auto position = static_cast<std::size_t>(index);
        const auto width = static_cast<double>(edge_data[position + 1]) -
                           static_cast<double>(edge_data[position]);
        values[position] =
            width > 0.0
                ? static_cast<float>(static_cast<double>(values[position]) /
                                     (total * width))
                : 0.0F;
      },
      checked_work(1));
}

} // namespace

auto bincount(const nd_array<std::int32_t> &values, int minimum_length)
    -> nd_array<std::int32_t> {
  if (minimum_length < 0)
    throw std::invalid_argument("bincount: minimum_length must be >= 0");
  const auto maximum = maximum_nonnegative(values);
  if (maximum == std::numeric_limits<int>::max())
    throw std::length_error("bincount: output is too large");
  const auto bin_count = std::max(maximum + 1, minimum_length);
  auto counts = count_binned<std::int32_t>(
      values, nullptr, bin_count, [](std::int32_t value) { return value; });
  return nd_array<std::int32_t>::from_buffer(std::move(counts), {bin_count});
}

auto bincount(const nd_array<std::int32_t> &values,
              const nd_array<float> &weights, int minimum_length)
    -> nd_array<float> {
  if (minimum_length < 0)
    throw std::invalid_argument("bincount: minimum_length must be >= 0");
  require_weights(values, weights, "bincount");
  const auto maximum = maximum_nonnegative(values);
  if (maximum == std::numeric_limits<int>::max())
    throw std::length_error("bincount: output is too large");
  const auto bin_count = std::max(maximum + 1, minimum_length);
  auto counts = count_binned<float>(values, &weights, bin_count,
                                    [](std::int32_t value) { return value; });
  return nd_array<float>::from_buffer(std::move(counts), {bin_count});
}

auto histogram_equal_width(const nd_array<float> &values, int bin_count,
                           float lower, float upper) -> histogram_result_int {
  if (bin_count <= 0)
    throw std::invalid_argument("histogram: bins must be > 0");
  const auto bounds = resolve_range(values, lower, upper);
  auto counts = count_equal_width<std::int32_t>(values, nullptr, bin_count,
                                                bounds.first, bounds.second);
  return {nd_array<std::int32_t>::from_buffer(std::move(counts), {bin_count}),
          linear_edges(bounds.first, bounds.second, bin_count)};
}

auto histogram_equal_width(const nd_array<float> &values,
                           const nd_array<float> &weights, int bin_count,
                           float lower, float upper) -> histogram_result_float {
  if (bin_count <= 0)
    throw std::invalid_argument("histogram: bins must be > 0");
  require_weights(values, weights, "histogram");
  const auto bounds = resolve_range(values, lower, upper);
  auto counts = count_equal_width<float>(values, &weights, bin_count,
                                         bounds.first, bounds.second);
  return {nd_array<float>::from_buffer(std::move(counts), {bin_count}),
          linear_edges(bounds.first, bounds.second, bin_count)};
}

auto histogram_edges(const nd_array<float> &values,
                     const nd_array<float> &edges) -> histogram_result_int {
  const auto bin_count = require_edges(edges);
  auto counts = count_edges<std::int32_t>(values, nullptr, edges, bin_count);
  return {nd_array<std::int32_t>::from_buffer(std::move(counts), {bin_count}),
          edges.shallow_copy()};
}

auto histogram_edges(const nd_array<float> &values,
                     const nd_array<float> &weights,
                     const nd_array<float> &edges) -> histogram_result_float {
  const auto bin_count = require_edges(edges);
  require_weights(values, weights, "histogram");
  auto counts = count_edges<float>(values, &weights, edges, bin_count);
  return {nd_array<float>::from_buffer(std::move(counts), {bin_count}),
          edges.shallow_copy()};
}

auto histogram_density_equal_width(const nd_array<float> &values, int bin_count,
                                   float lower, float upper,
                                   std::optional<nd_array<float>> weights)
    -> histogram_result_float {
  if (bin_count <= 0)
    throw std::invalid_argument("histogram: bins must be > 0");
  if (weights)
    require_weights(values, *weights, "histogram");
  const auto bounds = resolve_range(values, lower, upper);
  auto count_buffer =
      count_equal_width<float>(values, weights ? &*weights : nullptr, bin_count,
                               bounds.first, bounds.second);
  auto counts =
      nd_array<float>::from_buffer(std::move(count_buffer), {bin_count});
  auto edges = linear_edges(bounds.first, bounds.second, bin_count);
  apply_density(counts, edges);
  return {std::move(counts), std::move(edges)};
}

auto histogram_density_edges(const nd_array<float> &values,
                             const nd_array<float> &edges,
                             std::optional<nd_array<float>> weights)
    -> histogram_result_float {
  const auto bin_count = require_edges(edges);
  if (weights)
    require_weights(values, *weights, "histogram");
  auto count_buffer = count_edges<float>(values, weights ? &*weights : nullptr,
                                         edges, bin_count);
  auto counts =
      nd_array<float>::from_buffer(std::move(count_buffer), {bin_count});
  apply_density(counts, edges);
  return {std::move(counts), edges.shallow_copy()};
}

} // namespace tf::cpp
