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
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/geometry/triangulate.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

/// An array input names no arity of its own, so its axes are explicit here as
/// they are in the synchronous entry. What an array overload carries is a
/// shallow owning handle, so a mutation through any alias of that storage may
/// be observed by queued work.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          typename Resolver>
auto triangulate(Resolver &&resolver,
                 const cpp::mesh<Index, Real, Dims, Ngon> &value) {
  return submit<tf::polygons_buffer<Index, Real, Dims, 3>>(
      std::forward<Resolver>(resolver),
      [value] { return cpp::triangulate<Index, Real, Dims, Ngon>(value); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto triangulate(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, 3>> {
  return async::triangulate<Index, Real, Dims, Ngon>(future_resolver{}, value);
}

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto triangulate(Resolver &&resolver,
                 const offset_blocked_buffer<Index, Index> &faces,
                 const nd_array<Real> &points) {
  return submit<tf::polygons_buffer<Index, Real, Dims, 3>>(
      std::forward<Resolver>(resolver), [faces, points] {
        return cpp::triangulate<Index, Real, Dims>(faces, points);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate(const offset_blocked_buffer<Index, Index> &faces,
                 const nd_array<Real> &points)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, 3>> {
  return async::triangulate<Index, Real, Dims>(future_resolver{}, faces,
                                               points);
}

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto triangulate(Resolver &&resolver, const nd_array<Index> &faces,
                 const nd_array<Real> &points) {
  return submit<tf::polygons_buffer<Index, Real, Dims, 3>>(
      std::forward<Resolver>(resolver), [faces, points] {
        return cpp::triangulate<Index, Real, Dims>(faces, points);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate(const nd_array<Index> &faces, const nd_array<Real> &points)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, 3>> {
  return async::triangulate<Index, Real, Dims>(future_resolver{}, faces,
                                               points);
}

template <typename Index, typename Real, std::size_t Dims, typename Resolver>
auto triangulate(Resolver &&resolver, const nd_array<Real> &polygons)
    -> resolver_result_t<Resolver, tf::polygons_buffer<Index, Real, Dims, 3>> {
  return submit<tf::polygons_buffer<Index, Real, Dims, 3>>(
      std::forward<Resolver>(resolver),
      [polygons] { return cpp::triangulate<Index, Real, Dims>(polygons); });
}

template <typename Index, typename Real, std::size_t Dims>
auto triangulate(const nd_array<Real> &polygons)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, 3>> {
  return async::triangulate<Index, Real, Dims>(future_resolver{}, polygons);
}

} // namespace tf::cpp::async
