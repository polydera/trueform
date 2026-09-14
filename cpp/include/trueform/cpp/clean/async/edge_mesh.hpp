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

#include "trueform/core/segments_buffer.hpp"
#include "trueform/cpp/clean/edge_mesh.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto cleaned_edge_mesh(Resolver &&resolver,
                       const edge_mesh<Index, Real, Dims> &value,
                       Real tolerance = Real{},
                       bool remove_duplicate_primitives = true,
                       bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver, tf::segments_buffer<Index, Real, Dims>> {
  return submit<tf::segments_buffer<Index, Real, Dims>>(
      std::forward<Resolver>(resolver),
      [value, tolerance, remove_duplicate_primitives,
       remove_unreferenced_points] {
        return cpp::cleaned_edge_mesh<Index, Real, Dims>(
            value, tolerance, remove_duplicate_primitives,
            remove_unreferenced_points);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto cleaned_edge_mesh(const edge_mesh<Index, Real, Dims> &value,
                       Real tolerance = Real{},
                       bool remove_duplicate_primitives = true,
                       bool remove_unreferenced_points = true)
    -> std::future<tf::segments_buffer<Index, Real, Dims>> {
  return async::cleaned_edge_mesh<Index, Real, Dims>(
      future_resolver{}, value, tolerance, remove_duplicate_primitives,
      remove_unreferenced_points);
}

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto cleaned_edge_mesh_with_maps(Resolver &&resolver,
                                 const edge_mesh<Index, Real, Dims> &value,
                                 Real tolerance = Real{},
                                 bool remove_duplicate_primitives = true,
                                 bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver,
                         cleaned_edge_mesh_result<Index, Real, Dims>> {
  using result_type = cleaned_edge_mesh_result<Index, Real, Dims>;
  return submit<result_type>(
      std::forward<Resolver>(resolver),
      [value, tolerance, remove_duplicate_primitives,
       remove_unreferenced_points] {
        return cpp::cleaned_edge_mesh_with_maps<Index, Real, Dims>(
            value, tolerance, remove_duplicate_primitives,
            remove_unreferenced_points);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto cleaned_edge_mesh_with_maps(const edge_mesh<Index, Real, Dims> &value,
                                 Real tolerance = Real{},
                                 bool remove_duplicate_primitives = true,
                                 bool remove_unreferenced_points = true)
    -> std::future<cleaned_edge_mesh_result<Index, Real, Dims>> {
  return async::cleaned_edge_mesh_with_maps<Index, Real, Dims>(
      future_resolver{}, value, tolerance, remove_duplicate_primitives,
      remove_unreferenced_points);
}

} // namespace tf::cpp::async
