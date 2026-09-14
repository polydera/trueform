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

#include "trueform/cpp/clean/soup.hpp"
#include "trueform/cpp/clean/supported.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Index = default_index_t, typename Real, std::size_t Dims = 3,
          std::size_t Vertices = 3, typename Resolver,
          std::enable_if_t<
              is_supported_cleaned_polygon_soup_v<Index, Real, Dims, Vertices>,
              int> = 0>
auto cleaned_polygon_soup(Resolver &&resolver, const nd_array<Real> &polygons,
                          Real tolerance = Real{},
                          bool remove_duplicate_primitives = true,
                          bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver, basic_cleaned_polygon_soup_result<
                                       Index, Real, Dims, Vertices>> {
  using result_type =
      basic_cleaned_polygon_soup_result<Index, Real, Dims, Vertices>;
  return submit<result_type>(
      std::forward<Resolver>(resolver),
      [polygons, tolerance, remove_duplicate_primitives,
       remove_unreferenced_points] {
        return cpp::cleaned_polygon_soup<Index, Real, Dims, Vertices>(
            polygons, tolerance, remove_duplicate_primitives,
            remove_unreferenced_points);
      });
}

template <typename Index = default_index_t, typename Real, std::size_t Dims = 3,
          std::size_t Vertices = 3,
          std::enable_if_t<
              is_supported_cleaned_polygon_soup_v<Index, Real, Dims, Vertices>,
              int> = 0>
auto cleaned_polygon_soup(const nd_array<Real> &polygons,
                          Real tolerance = Real{},
                          bool remove_duplicate_primitives = true,
                          bool remove_unreferenced_points = true)
    -> std::future<
        basic_cleaned_polygon_soup_result<Index, Real, Dims, Vertices>> {
  return async::cleaned_polygon_soup<Index, Real, Dims, Vertices>(
      future_resolver{}, polygons, tolerance, remove_duplicate_primitives,
      remove_unreferenced_points);
}

} // namespace tf::cpp::async
