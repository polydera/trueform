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
#pragma once

#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>
#include <optional>

namespace tf::cpp {

struct histogram_result_int {
  nd_array<std::int32_t> counts;
  nd_array<float> edges;
};

struct histogram_result_float {
  nd_array<float> counts;
  nd_array<float> edges;
};

auto bincount(const nd_array<std::int32_t> &values, int minimum_length = 0)
    -> nd_array<std::int32_t>;

auto bincount(const nd_array<std::int32_t> &values,
              const nd_array<float> &weights, int minimum_length = 0)
    -> nd_array<float>;

auto histogram_equal_width(const nd_array<float> &values, int bin_count,
                           float lower, float upper) -> histogram_result_int;

auto histogram_equal_width(const nd_array<float> &values,
                           const nd_array<float> &weights, int bin_count,
                           float lower, float upper) -> histogram_result_float;

auto histogram_edges(const nd_array<float> &values,
                     const nd_array<float> &edges) -> histogram_result_int;

auto histogram_edges(const nd_array<float> &values,
                     const nd_array<float> &weights,
                     const nd_array<float> &edges) -> histogram_result_float;

auto histogram_density_equal_width(
    const nd_array<float> &values, int bin_count, float lower, float upper,
    std::optional<nd_array<float>> weights = std::nullopt)
    -> histogram_result_float;

auto histogram_density_edges(
    const nd_array<float> &values, const nd_array<float> &edges,
    std::optional<nd_array<float>> weights = std::nullopt)
    -> histogram_result_float;

} // namespace tf::cpp
