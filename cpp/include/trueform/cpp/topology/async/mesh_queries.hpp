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
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/topology/boundary_curves.hpp"
#include "trueform/cpp/topology/boundary_edges.hpp"
#include "trueform/cpp/topology/boundary_paths.hpp"
#include "trueform/cpp/topology/euler_characteristic.hpp"
#include "trueform/cpp/topology/is_closed.hpp"
#include "trueform/cpp/topology/is_manifold.hpp"
#include "trueform/cpp/topology/non_manifold_edges.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {
/// The async form of every topology question that takes nothing but a mesh:
/// one mechanism, stated once and applied to the operations that share it.
///
/// A mesh is already one coherent reading, so it is carried to the executor as
/// it stands; the arrays and the cache behind it are the caller's, and the
/// caller keeps them alive for as long as the call is pending.
#define TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(Operation, ...)                      \
  template <typename Resolver, typename Index, typename Real,                  \
            std::size_t Dims, std::size_t Ngon>                                \
  auto Operation(Resolver &&resolver,                                          \
                 const cpp::mesh<Index, Real, Dims, Ngon> &value)              \
      -> resolver_result_t<Resolver, __VA_ARGS__> {                            \
    return submit<__VA_ARGS__>(std::forward<Resolver>(resolver),               \
                               [value] { return cpp::Operation(value); });     \
  }                                                                            \
                                                                               \
  template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon> \
  auto Operation(const cpp::mesh<Index, Real, Dims, Ngon> &value)              \
      -> std::future<__VA_ARGS__> {                                            \
    return async::Operation(future_resolver{}, value);                         \
  }

TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(is_closed, bool)
TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(is_open, bool)
TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(is_manifold, bool)
TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(is_non_manifold, bool)
TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(euler_characteristic, std::int32_t)
TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(non_manifold_edges, nd_array<Index>)
TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(boundary_edges, nd_array<Index>)
TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(boundary_paths,
                                  offset_blocked_buffer<Index, Index>)
TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY(boundary_curves,
                                  boundary_curves_result<Index, Real, Dims>)

#undef TF_CPP_DEFINE_ASYNC_MESH_TOPOLOGY

} // namespace tf::cpp::async
