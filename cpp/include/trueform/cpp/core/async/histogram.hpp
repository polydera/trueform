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

#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/histogram.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstdint>
#include <future>
#include <optional>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver>
auto bincount(Resolver &&resolver, const nd_array<std::int32_t> &values,
              int minimum_length = 0)
    -> resolver_result_t<Resolver, nd_array<std::int32_t>> {
  return submit<nd_array<std::int32_t>>(
      std::forward<Resolver>(resolver), [values, minimum_length] {
        return cpp::bincount(values, minimum_length);
      });
}

inline auto bincount(const nd_array<std::int32_t> &values,
                     int minimum_length = 0)
    -> std::future<nd_array<std::int32_t>> {
  return async::bincount(future_resolver{}, values, minimum_length);
}

template <typename Resolver>
auto bincount(Resolver &&resolver, const nd_array<std::int32_t> &values,
              const nd_array<float> &weights, int minimum_length = 0)
    -> resolver_result_t<Resolver, nd_array<float>> {
  return submit<nd_array<float>>(
      std::forward<Resolver>(resolver), [values, weights, minimum_length] {
        return cpp::bincount(values, weights, minimum_length);
      });
}

inline auto bincount(const nd_array<std::int32_t> &values,
                     const nd_array<float> &weights, int minimum_length = 0)
    -> std::future<nd_array<float>> {
  return async::bincount(future_resolver{}, values, weights, minimum_length);
}

template <typename Resolver>
auto histogram_equal_width(Resolver &&resolver, const nd_array<float> &values,
                           int bin_count, float lower, float upper)
    -> resolver_result_t<Resolver, histogram_result_int> {
  return submit<histogram_result_int>(
      std::forward<Resolver>(resolver), [values, bin_count, lower, upper] {
        return cpp::histogram_equal_width(values, bin_count, lower, upper);
      });
}

inline auto histogram_equal_width(const nd_array<float> &values, int bin_count,
                                  float lower, float upper)
    -> std::future<histogram_result_int> {
  return async::histogram_equal_width(future_resolver{}, values, bin_count,
                                      lower, upper);
}

template <typename Resolver>
auto histogram_equal_width(Resolver &&resolver, const nd_array<float> &values,
                           const nd_array<float> &weights, int bin_count,
                           float lower, float upper)
    -> resolver_result_t<Resolver, histogram_result_float> {
  return submit<histogram_result_float>(
      std::forward<Resolver>(resolver),
      [values, weights, bin_count, lower, upper] {
        return cpp::histogram_equal_width(values, weights, bin_count, lower,
                                          upper);
      });
}

inline auto histogram_equal_width(const nd_array<float> &values,
                                  const nd_array<float> &weights, int bin_count,
                                  float lower, float upper)
    -> std::future<histogram_result_float> {
  return async::histogram_equal_width(future_resolver{}, values, weights,
                                      bin_count, lower, upper);
}

template <typename Resolver>
auto histogram_edges(Resolver &&resolver, const nd_array<float> &values,
                     const nd_array<float> &edges)
    -> resolver_result_t<Resolver, histogram_result_int> {
  return submit<histogram_result_int>(
      std::forward<Resolver>(resolver),
      [values, edges] { return cpp::histogram_edges(values, edges); });
}

inline auto histogram_edges(const nd_array<float> &values,
                            const nd_array<float> &edges)
    -> std::future<histogram_result_int> {
  return async::histogram_edges(future_resolver{}, values, edges);
}

template <typename Resolver>
auto histogram_edges(Resolver &&resolver, const nd_array<float> &values,
                     const nd_array<float> &weights,
                     const nd_array<float> &edges)
    -> resolver_result_t<Resolver, histogram_result_float> {
  return submit<histogram_result_float>(
      std::forward<Resolver>(resolver), [values, weights, edges] {
        return cpp::histogram_edges(values, weights, edges);
      });
}

inline auto histogram_edges(const nd_array<float> &values,
                            const nd_array<float> &weights,
                            const nd_array<float> &edges)
    -> std::future<histogram_result_float> {
  return async::histogram_edges(future_resolver{}, values, weights, edges);
}

template <typename Resolver>
auto histogram_density_equal_width(
    Resolver &&resolver, const nd_array<float> &values, int bin_count,
    float lower, float upper,
    std::optional<nd_array<float>> weights = std::nullopt)
    -> resolver_result_t<Resolver, histogram_result_float> {
  return submit<histogram_result_float>(
      std::forward<Resolver>(resolver), [values, weights = std::move(weights),
                                         bin_count, lower, upper]() mutable {
        return cpp::histogram_density_equal_width(values, bin_count, lower,
                                                  upper, std::move(weights));
      });
}

inline auto histogram_density_equal_width(
    const nd_array<float> &values, int bin_count, float lower, float upper,
    std::optional<nd_array<float>> weights = std::nullopt)
    -> std::future<histogram_result_float> {
  return async::histogram_density_equal_width(
      future_resolver{}, values, bin_count, lower, upper, std::move(weights));
}

template <typename Resolver>
auto histogram_density_edges(
    Resolver &&resolver, const nd_array<float> &values,
    const nd_array<float> &edges,
    std::optional<nd_array<float>> weights = std::nullopt)
    -> resolver_result_t<Resolver, histogram_result_float> {
  return submit<histogram_result_float>(
      std::forward<Resolver>(resolver),
      [values, edges, weights = std::move(weights)]() mutable {
        return cpp::histogram_density_edges(values, edges, std::move(weights));
      });
}

inline auto
histogram_density_edges(const nd_array<float> &values,
                        const nd_array<float> &edges,
                        std::optional<nd_array<float>> weights = std::nullopt)
    -> std::future<histogram_result_float> {
  return async::histogram_density_edges(future_resolver{}, values, edges,
                                        std::move(weights));
}

} // namespace tf::cpp::async
