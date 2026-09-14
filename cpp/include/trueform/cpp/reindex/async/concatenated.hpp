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
#include "trueform/cpp/core/common_index.hpp"
#include "trueform/cpp/core/detail/concatenated_arity.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/reindex/concatenated.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto concatenate_meshes(
    Resolver &&resolver,
    const std::vector<mesh<Index, Real, Dims, Ngon>> &meshes) {
  return submit<tf::polygons_buffer<Index, Real, Dims, Ngon>>(
      std::forward<Resolver>(resolver), [meshes] {
        return cpp::concatenate_meshes<Index, Real, Dims, Ngon>(meshes);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto concatenate_meshes(
    const std::vector<mesh<Index, Real, Dims, Ngon>> &meshes)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  return async::concatenate_meshes<future_resolver, Index, Real, Dims, Ngon>(
      future_resolver{}, meshes);
}

template <typename Resolver, typename Index, typename Real, std::size_t Dims>
auto concatenate_edge_meshes(
    Resolver &&resolver,
    const std::vector<edge_mesh<Index, Real, Dims>> &meshes) {
  return submit<tf::segments_buffer<Index, Real, Dims>>(
      std::forward<Resolver>(resolver), [meshes] {
        return cpp::concatenate_edge_meshes<Index, Real, Dims>(meshes);
      });
}

template <typename Index, typename Real, std::size_t Dims>
auto concatenate_edge_meshes(
    const std::vector<edge_mesh<Index, Real, Dims>> &meshes)
    -> std::future<tf::segments_buffer<Index, Real, Dims>> {
  return async::concatenate_edge_meshes<future_resolver, Index, Real, Dims>(
      future_resolver{}, meshes);
}

template <typename Resolver, typename Index0, typename Real0, typename Index1,
          typename Real1, std::size_t Dims, std::size_t Ngon0,
          std::size_t Ngon1>
auto concatenate_meshes(Resolver &&resolver,
                        const mesh<Index0, Real0, Dims, Ngon0> &first,
                        const mesh<Index1, Real1, Dims, Ngon1> &second) {
  using result_type =
      tf::polygons_buffer<common_index_t<Index0, Index1>,
                          std::common_type_t<Real0, Real1>, Dims,
                          cpp::detail::concatenated_arity_v<Ngon0, Ngon1>>;
  return submit<result_type>(
      std::forward<Resolver>(resolver), [first, second] {
        return cpp::concatenate_meshes<Index0, Real0, Index1, Real1, Dims,
                                       Ngon0, Ngon1>(first, second);
      });
}

template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>
auto concatenate_meshes(const mesh<Index0, Real0, Dims, Ngon0> &first,
                        const mesh<Index1, Real1, Dims, Ngon1> &second)
    -> std::future<tf::polygons_buffer<
        common_index_t<Index0, Index1>, std::common_type_t<Real0, Real1>, Dims,
        cpp::detail::concatenated_arity_v<Ngon0, Ngon1>>> {
  return async::concatenate_meshes(future_resolver{}, first, second);
}

template <typename Resolver, typename Index0, typename Real0, typename Index1,
          typename Real1, std::size_t Dims>
auto concatenate_edge_meshes(
    Resolver &&resolver, const edge_mesh<Index0, Real0, Dims> &first,
    const edge_mesh<Index1, Real1, Dims> &second) {
  using result_type = tf::segments_buffer<common_index_t<Index0, Index1>,
                                          std::common_type_t<Real0, Real1>,
                                          Dims>;
  return submit<result_type>(
      std::forward<Resolver>(resolver), [first, second] {
        return cpp::concatenate_edge_meshes<Index0, Real0, Index1, Real1, Dims>(
            first, second);
      });
}

template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims>
auto concatenate_edge_meshes(
    const edge_mesh<Index0, Real0, Dims> &first,
    const edge_mesh<Index1, Real1, Dims> &second)
    -> std::future<tf::segments_buffer<common_index_t<Index0, Index1>,
                                       std::common_type_t<Real0, Real1>,
                                       Dims>> {
  return async::concatenate_edge_meshes(future_resolver{}, first, second);
}

} // namespace tf::cpp::async
