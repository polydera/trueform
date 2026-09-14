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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/geometry/make_tube_mesh.hpp"

#include <cstdint>
#include <future>
#include <utility>

namespace tf::cpp::async {

/// @brief Resolve tube creation on the common executor.
template <typename Resolver, typename Index, typename Real>
auto make_tube_mesh(Resolver &&resolver,
                    const offset_blocked_buffer<Index, Index> &paths,
                    const nd_array<Real> &points, Real radius,
                    std::int32_t radial_segments = 8) {
  return submit<cpp::detail::minted_mesh_result_t<Index, Real, 3>>(
      std::forward<Resolver>(resolver),
      [paths, points, radius, radial_segments] {
        return cpp::make_tube_mesh(paths, points, radius, radial_segments);
      });
}

template <typename Index, typename Real>
auto make_tube_mesh(const offset_blocked_buffer<Index, Index> &paths,
                    const nd_array<Real> &points, Real radius,
                    std::int32_t radial_segments = 8)
    -> std::future<cpp::detail::minted_mesh_result_t<Index, Real, 3>> {
  return async::make_tube_mesh(future_resolver{}, paths, points, radius,
                               radial_segments);
}

} // namespace tf::cpp::async
