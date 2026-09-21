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

#include "trueform/core/curves_buffer.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/common_index.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/intersect/intersection_curves.hpp"
#include "trueform/intersect/intersect_config.hpp"

#include <cstddef>
#include <future>
#include <utility>
#include <vector>

namespace tf::cpp::async {

template <typename Resolver, typename Index0, typename Real, typename Index1,
          std::size_t Ngon0, std::size_t Ngon1>
auto intersection_curves(Resolver &&resolver,
                         const mesh<Index0, Real, 3, Ngon0> &a,
                         const mesh<Index1, Real, 3, Ngon1> &b,
                         tf::intersect_config config = {})
    -> resolver_result_t<
        Resolver, tf::curves_buffer<common_index_t<Index0, Index1>, Real, 3>> {
  return submit<tf::curves_buffer<common_index_t<Index0, Index1>, Real, 3>>(
      std::forward<Resolver>(resolver),
      [a, b, config] { return cpp::intersection_curves(a, b, config); });
}

template <typename Index0, typename Real, typename Index1, std::size_t Ngon0,
          std::size_t Ngon1>
auto intersection_curves(const mesh<Index0, Real, 3, Ngon0> &a,
                         const mesh<Index1, Real, 3, Ngon1> &b,
                         tf::intersect_config config = {})
    -> std::future<tf::curves_buffer<common_index_t<Index0, Index1>, Real, 3>> {
  return async::intersection_curves(future_resolver{}, a, b, config);
}

template <typename Resolver, typename Index, typename Real, std::size_t Ngon>
auto intersection_curves(Resolver &&resolver,
                         const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
                         tf::intersect_config config = {})
    -> resolver_result_t<Resolver, tf::curves_buffer<Index, Real, 3>> {
  return submit<tf::curves_buffer<Index, Real, 3>>(
      std::forward<Resolver>(resolver),
      [meshes, config] { return cpp::intersection_curves(meshes, config); });
}

template <typename Index, typename Real, std::size_t Ngon>
auto intersection_curves(const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,
                         tf::intersect_config config = {})
    -> std::future<tf::curves_buffer<Index, Real, 3>> {
  return async::intersection_curves(future_resolver{}, meshes, config);
}

} // namespace tf::cpp::async
