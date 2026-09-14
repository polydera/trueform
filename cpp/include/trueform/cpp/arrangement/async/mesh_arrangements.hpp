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
#include "trueform/cpp/arrangement/mesh_arrangements.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>
#include <future>
#include <utility>
#include <vector>

namespace tf::cpp::async {

#define TF_CPP_DEFINE_ASYNC_MESH_ARRANGEMENTS(Operation, Result)               \
  template <typename Resolver, typename Index, typename Real,                  \
            std::size_t Ngon>                                                  \
  auto Operation(Resolver &&resolver,                                          \
                 const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,        \
                 tf::arrangement_config config = {})                           \
      -> resolver_result_t<Resolver, Result<Index, Real, Ngon>> {              \
    return submit<Result<Index, Real, Ngon>>(                                  \
        std::forward<Resolver>(resolver),                                      \
        [meshes, config] { return cpp::Operation(meshes, config); });          \
  }                                                                            \
                                                                               \
  template <typename Index, typename Real, std::size_t Ngon>                   \
  auto Operation(const std::vector<mesh<Index, Real, 3, Ngon>> &meshes,        \
                 tf::arrangement_config config = {})                           \
      -> std::future<Result<Index, Real, Ngon>> {                              \
    return async::Operation(future_resolver{}, meshes, config);                \
  }

TF_CPP_DEFINE_ASYNC_MESH_ARRANGEMENTS(mesh_arrangements,
                                      mesh_arrangement_result)
TF_CPP_DEFINE_ASYNC_MESH_ARRANGEMENTS(mesh_arrangements_with_curves,
                                      mesh_arrangement_with_curves_result)

#undef TF_CPP_DEFINE_ASYNC_MESH_ARRANGEMENTS

} // namespace tf::cpp::async
