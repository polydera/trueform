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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/segments_buffer.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/reindex/split_into_components.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto split_into_components(Resolver &&resolver,
                           const mesh<Index, Real, Dims, Ngon> &value,
                           const nd_array<std::int32_t> &labels) {
  using result_type =
      split_components_result<tf::polygons_buffer<Index, Real, Dims, Ngon>>;
  return submit<result_type>(std::forward<Resolver>(resolver), [value, labels] {
    return cpp::split_into_components<Index, Real, Dims, Ngon>(value, labels);
  });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto split_into_components(const mesh<Index, Real, Dims, Ngon> &value,
                           const nd_array<std::int32_t> &labels)
    -> std::future<
        split_components_result<tf::polygons_buffer<Index, Real, Dims, Ngon>>> {
  return async::split_into_components<future_resolver, Index, Real, Dims, Ngon>(
      future_resolver{}, value, labels);
}

template <typename Resolver, typename Index, typename Real, std::size_t Dims>
auto split_into_components(Resolver &&resolver,
                           const edge_mesh<Index, Real, Dims> &value,
                           const nd_array<std::int32_t> &labels) {
  using result_type =
      split_components_result<tf::segments_buffer<Index, Real, Dims>>;
  return submit<result_type>(std::forward<Resolver>(resolver), [value, labels] {
    return cpp::split_into_components<Index, Real, Dims>(value, labels);
  });
}

template <typename Index, typename Real, std::size_t Dims>
auto split_into_components(const edge_mesh<Index, Real, Dims> &value,
                           const nd_array<std::int32_t> &labels)
    -> std::future<
        split_components_result<tf::segments_buffer<Index, Real, Dims>>> {
  return async::split_into_components<future_resolver, Index, Real, Dims>(
      future_resolver{}, value, labels);
}

} // namespace tf::cpp::async
