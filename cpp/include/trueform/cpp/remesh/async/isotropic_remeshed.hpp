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
#include "trueform/cpp/remesh/isotropic_remeshed.hpp"
#include "trueform/remesh/isotropic_remesh_config.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index, typename Real>
auto isotropic_remeshed(Resolver &&resolver, const mesh<Index, Real, 3> &value,
                        tf::isotropic_remesh_config<Real> config,
                        const nd_array<std::int32_t> &regions = {})
    -> resolver_result_t<Resolver, remesh_result<Index, Real>> {
  using result_type = remesh_result<Index, Real>;
  return submit<result_type>(
      std::forward<Resolver>(resolver), [value, config, regions] {
        return cpp::isotropic_remeshed<Index, Real>(value, config, regions);
      });
}

template <typename Index, typename Real>
auto isotropic_remeshed(const mesh<Index, Real, 3> &value,
                        tf::isotropic_remesh_config<Real> config,
                        const nd_array<std::int32_t> &regions = {})
    -> std::future<remesh_result<Index, Real>> {
  return async::isotropic_remeshed<future_resolver, Index, Real>(
      future_resolver{}, value, config, regions);
}

} // namespace tf::cpp::async
