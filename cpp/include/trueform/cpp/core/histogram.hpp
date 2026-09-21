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

/// @brief Binned counts and the edges of the bins they were counted into.
///
/// `Count` is the width the counts are stated in: the integer width the caller
/// carries, or `float` wherever the increment is a weight and the accumulator
/// a float sum. An nd_array holds at most int-many elements, so no count needs
/// more than int32 — the width answers the caller's carrier, not a magnitude.
template <typename Count> struct histogram_result {
  nd_array<Count> counts;
  nd_array<float> edges;
};

/// @brief One bin per id, counted in the width the ids arrived in.
template <typename T>
auto bincount(const nd_array<T> &values, int minimum_length = 0) -> nd_array<T>;

template <typename T>
auto bincount(const nd_array<T> &values, const nd_array<float> &weights,
              int minimum_length = 0) -> nd_array<float>;

template <typename Count = std::int32_t>
auto histogram_equal_width(const nd_array<float> &values, int bin_count,
                           float lower, float upper) -> histogram_result<Count>;

auto histogram_equal_width(const nd_array<float> &values,
                           const nd_array<float> &weights, int bin_count,
                           float lower, float upper) -> histogram_result<float>;

template <typename Count = std::int32_t>
auto histogram_edges(const nd_array<float> &values,
                     const nd_array<float> &edges) -> histogram_result<Count>;

auto histogram_edges(const nd_array<float> &values,
                     const nd_array<float> &weights,
                     const nd_array<float> &edges) -> histogram_result<float>;

auto histogram_density_equal_width(
    const nd_array<float> &values, int bin_count, float lower, float upper,
    std::optional<nd_array<float>> weights = std::nullopt)
    -> histogram_result<float>;

auto histogram_density_edges(
    const nd_array<float> &values, const nd_array<float> &edges,
    std::optional<nd_array<float>> weights = std::nullopt)
    -> histogram_result<float>;

#define TF_CPP_EXTERN_HISTOGRAM(T)                                             \
  extern template auto bincount<T>(const nd_array<T> &, int) -> nd_array<T>;   \
  extern template auto bincount<T>(                                            \
      const nd_array<T> &, const nd_array<float> &, int) -> nd_array<float>;   \
  extern template auto histogram_equal_width<T>(                               \
      const nd_array<float> &, int, float, float) -> histogram_result<T>;      \
  extern template auto histogram_edges<T>(                                     \
      const nd_array<float> &, const nd_array<float> &) -> histogram_result<T>

TF_CPP_EXTERN_HISTOGRAM(std::int32_t);
TF_CPP_EXTERN_HISTOGRAM(std::int64_t);

#undef TF_CPP_EXTERN_HISTOGRAM

} // namespace tf::cpp
