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
#include "trueform/cpp/geometry/make_plane_mesh.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

/// @brief Resolve plane creation on the common executor.
template <typename Index = default_index_t, typename Real, typename Resolver>
auto make_plane_mesh(Resolver &&resolver, Real width, Real height,
                     std::int32_t width_ticks = 1,
                     std::int32_t height_ticks = 1)
    -> resolver_result_t<Resolver,
                         cpp::detail::minted_mesh_result_t<Index, Real, 3>> {
  using result_type = cpp::detail::minted_mesh_result_t<Index, Real, 3>;
  return submit<result_type>(std::forward<Resolver>(resolver),
                             [width, height, width_ticks, height_ticks] {
                               return cpp::make_plane_mesh<Index, Real>(
                                   width, height, width_ticks, height_ticks);
                             });
}

template <typename Index = default_index_t, typename Real>
auto make_plane_mesh(Real width, Real height, std::int32_t width_ticks = 1,
                     std::int32_t height_ticks = 1)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, Real, 3>> {
  return async::make_plane_mesh<Index, Real>(future_resolver{}, width, height,
                                             width_ticks, height_ticks);
}

} // namespace tf::cpp::async
