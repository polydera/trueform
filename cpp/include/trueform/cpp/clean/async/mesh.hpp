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
#include "trueform/cpp/clean/mesh.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          typename Resolver>
auto cleaned_mesh(Resolver &&resolver,
                  const mesh<Index, Real, Dims, Ngon> &value,
                  Real tolerance = Real{},
                  bool remove_duplicate_primitives = true,
                  bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver,
                         tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  return submit<tf::polygons_buffer<Index, Real, Dims, Ngon>>(
      std::forward<Resolver>(resolver),
      [value, tolerance, remove_duplicate_primitives,
       remove_unreferenced_points] {
        return cpp::cleaned_mesh<Index, Real, Dims, Ngon>(
            value, tolerance, remove_duplicate_primitives,
            remove_unreferenced_points);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cleaned_mesh(const mesh<Index, Real, Dims, Ngon> &value,
                  Real tolerance = Real{},
                  bool remove_duplicate_primitives = true,
                  bool remove_unreferenced_points = true)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  return async::cleaned_mesh<Index, Real, Dims, Ngon>(
      future_resolver{}, value, tolerance, remove_duplicate_primitives,
      remove_unreferenced_points);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          typename Resolver>
auto cleaned_mesh_with_maps(Resolver &&resolver,
                            const mesh<Index, Real, Dims, Ngon> &value,
                            Real tolerance = Real{},
                            bool remove_duplicate_primitives = true,
                            bool remove_unreferenced_points = true)
    -> resolver_result_t<Resolver,
                         cleaned_mesh_result<Index, Real, Dims, Ngon>> {
  using result_type = cleaned_mesh_result<Index, Real, Dims, Ngon>;
  return submit<result_type>(
      std::forward<Resolver>(resolver),
      [value, tolerance, remove_duplicate_primitives,
       remove_unreferenced_points] {
        return cpp::cleaned_mesh_with_maps<Index, Real, Dims, Ngon>(
            value, tolerance, remove_duplicate_primitives,
            remove_unreferenced_points);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cleaned_mesh_with_maps(const mesh<Index, Real, Dims, Ngon> &value,
                            Real tolerance = Real{},
                            bool remove_duplicate_primitives = true,
                            bool remove_unreferenced_points = true)
    -> std::future<cleaned_mesh_result<Index, Real, Dims, Ngon>> {
  return async::cleaned_mesh_with_maps<Index, Real, Dims, Ngon>(
      future_resolver{}, value, tolerance, remove_duplicate_primitives,
      remove_unreferenced_points);
}

} // namespace tf::cpp::async
