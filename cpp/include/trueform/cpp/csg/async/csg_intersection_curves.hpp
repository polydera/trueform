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
#include "trueform/cpp/csg/csg_graph.hpp"
#include "trueform/cpp/csg/csg_intersection_curves.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real>
auto csg_intersection_curves(Resolver &&resolver,
                             const csg_graph<Index, Real> &graph)
    -> resolver_result_t<Resolver, tf::curves_buffer<Index, Real, 3>> {
  auto owned_graph = graph;
  return submit<tf::curves_buffer<Index, Real, 3>>(
      std::forward<Resolver>(resolver), [graph = std::move(owned_graph)] {
        return cpp::csg_intersection_curves(graph);
      });
}

template <typename Index, typename Real>
auto csg_intersection_curves(const csg_graph<Index, Real> &graph)
    -> std::future<tf::curves_buffer<Index, Real, 3>> {
  return async::csg_intersection_curves(future_resolver{}, graph);
}

} // namespace tf::cpp::async
