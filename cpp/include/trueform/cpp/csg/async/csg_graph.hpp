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

#include "trueform/arrangement/arrangement_config.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/csg/csg_graph.hpp"

#include <cstdint>
#include <future>
#include <utility>
#include <vector>

namespace tf::cpp::async {

/// The graph outlives this call, and it reads its operands for as long as it
/// lives: the borrow law is the caller's, stated on the graph itself.
template <typename Resolver, typename Index, typename Real>
auto make_csg_graph(Resolver &&resolver,
                    const std::vector<mesh<Index, Real, 3, 3>> &meshes,
                    const std::vector<std::int32_t> &sheets = {},
                    tf::arrangement_config config = {})
    -> resolver_result_t<Resolver, csg_graph<Index, Real>> {
  return submit<csg_graph<Index, Real>>(
      std::forward<Resolver>(resolver), [meshes, sheets, config]() mutable {
        return cpp::make_csg_graph(std::move(meshes), std::move(sheets),
                                   config);
      });
}

template <typename Index, typename Real>
auto make_csg_graph(const std::vector<mesh<Index, Real, 3, 3>> &meshes,
                    const std::vector<std::int32_t> &sheets = {},
                    tf::arrangement_config config = {})
    -> std::future<csg_graph<Index, Real>> {
  return async::make_csg_graph(future_resolver{}, meshes, sheets, config);
}

} // namespace tf::cpp::async
