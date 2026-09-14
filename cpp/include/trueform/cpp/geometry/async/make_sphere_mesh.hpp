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
#include "trueform/cpp/core/detail/minted_mesh_result.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

/// @brief Resolve sphere creation on the common executor.
template <typename Index = default_index_t, typename Real, typename Resolver>
auto make_sphere_mesh(Resolver &&resolver, Real radius, std::int32_t stacks,
                      std::int32_t segments)
    -> resolver_result_t<Resolver,
                         cpp::detail::minted_mesh_result_t<Index, Real, 3>> {
  using result_type = cpp::detail::minted_mesh_result_t<Index, Real, 3>;
  return submit<result_type>(
      std::forward<Resolver>(resolver), [radius, stacks, segments] {
        return cpp::make_sphere_mesh<Index, Real>(radius, stacks, segments);
      });
}

template <typename Index = default_index_t, typename Real>
auto make_sphere_mesh(Real radius, std::int32_t stacks, std::int32_t segments)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, Real, 3>> {
  return async::make_sphere_mesh<Index, Real>(future_resolver{}, radius, stacks,
                                              segments);
}

} // namespace tf::cpp::async
