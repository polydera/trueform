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
#include "trueform/cpp/csg/csg_created_points.hpp"
#include "trueform/cpp/csg/csg_graph.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real>
auto csg_created_points(Resolver &&resolver,
                        const csg_graph<Index, Real> &graph)
    -> resolver_result_t<Resolver, nd_array<Real>> {
  auto owned_graph = graph;
  return submit<nd_array<Real>>(std::forward<Resolver>(resolver),
                                [graph = std::move(owned_graph)] {
                                  return cpp::csg_created_points(graph);
                                });
}

template <typename Index, typename Real>
auto csg_created_points(const csg_graph<Index, Real> &graph)
    -> std::future<nd_array<Real>> {
  return async::csg_created_points(future_resolver{}, graph);
}

} // namespace tf::cpp::async
