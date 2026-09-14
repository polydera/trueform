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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/topology/cdt.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Real, typename Resolver>
auto make_cdt(Resolver &&resolver, const nd_array<Real> &points)
    -> resolver_result_t<Resolver, cdt_result<Real>> {
  return submit<cdt_result<Real>>(std::forward<Resolver>(resolver),
                                  [points] { return cpp::make_cdt(points); });
}

template <typename Real>
auto make_cdt(const nd_array<Real> &points) -> std::future<cdt_result<Real>> {
  return async::make_cdt<Real>(future_resolver{}, points);
}

template <typename Real, typename Resolver>
auto make_cdt_with_maps(Resolver &&resolver, const nd_array<Real> &points)
    -> resolver_result_t<Resolver, cdt_result_with_map<Real>> {
  return submit<cdt_result_with_map<Real>>(
      std::forward<Resolver>(resolver),
      [points] { return cpp::make_cdt_with_maps(points); });
}

template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points)
    -> std::future<cdt_result_with_map<Real>> {
  return async::make_cdt_with_maps<Real>(future_resolver{}, points);
}

template <typename Real, typename Resolver>
auto make_cdt(Resolver &&resolver, const nd_array<Real> &points,
              const nd_array<std::int32_t> &edges,
              bool split_constraints = true)
    -> resolver_result_t<Resolver, cdt_result<Real>> {
  return submit<cdt_result<Real>>(
      std::forward<Resolver>(resolver), [points, edges, split_constraints] {
        return cpp::make_cdt(points, edges, split_constraints);
      });
}

template <typename Real>
auto make_cdt(const nd_array<Real> &points, const nd_array<std::int32_t> &edges,
              bool split_constraints = true) -> std::future<cdt_result<Real>> {
  return async::make_cdt<Real>(future_resolver{}, points, edges,
                               split_constraints);
}

template <typename Real, typename Resolver>
auto make_cdt_with_maps(Resolver &&resolver, const nd_array<Real> &points,
                        const nd_array<std::int32_t> &edges,
                        bool split_constraints = true)
    -> resolver_result_t<Resolver, cdt_result_with_map<Real>> {
  return submit<cdt_result_with_map<Real>>(
      std::forward<Resolver>(resolver), [points, edges, split_constraints] {
        return cpp::make_cdt_with_maps(points, edges, split_constraints);
      });
}

template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points,
                        const nd_array<std::int32_t> &edges,
                        bool split_constraints = true)
    -> std::future<cdt_result_with_map<Real>> {
  return async::make_cdt_with_maps<Real>(future_resolver{}, points, edges,
                                         split_constraints);
}

template <typename Real, typename Resolver>
auto make_cdt(Resolver &&resolver, const nd_array<Real> &points,
              const nd_array<std::int32_t> &edges,
              const nd_array<std::int8_t> &edge_mask,
              bool split_constraints = true)
    -> resolver_result_t<Resolver, cdt_result<Real>> {
  return submit<cdt_result<Real>>(
      std::forward<Resolver>(resolver),
      [points, edges, edge_mask, split_constraints] {
        return cpp::make_cdt(points, edges, edge_mask, split_constraints);
      });
}

template <typename Real>
auto make_cdt(const nd_array<Real> &points, const nd_array<std::int32_t> &edges,
              const nd_array<std::int8_t> &edge_mask,
              bool split_constraints = true) -> std::future<cdt_result<Real>> {
  return async::make_cdt<Real>(future_resolver{}, points, edges, edge_mask,
                               split_constraints);
}

template <typename Real, typename Resolver>
auto make_cdt_with_maps(Resolver &&resolver, const nd_array<Real> &points,
                        const nd_array<std::int32_t> &edges,
                        const nd_array<std::int8_t> &edge_mask,
                        bool split_constraints = true)
    -> resolver_result_t<Resolver, cdt_result_with_map<Real>> {
  return submit<cdt_result_with_map<Real>>(
      std::forward<Resolver>(resolver),
      [points, edges, edge_mask, split_constraints] {
        return cpp::make_cdt_with_maps(points, edges, edge_mask,
                                       split_constraints);
      });
}

template <typename Real>
auto make_cdt_with_maps(const nd_array<Real> &points,
                        const nd_array<std::int32_t> &edges,
                        const nd_array<std::int8_t> &edge_mask,
                        bool split_constraints = true)
    -> std::future<cdt_result_with_map<Real>> {
  return async::make_cdt_with_maps<Real>(future_resolver{}, points, edges,
                                         edge_mask, split_constraints);
}

} // namespace tf::cpp::async
