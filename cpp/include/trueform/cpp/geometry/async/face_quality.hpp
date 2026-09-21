/*
 * Copyright (c) 2026 XLAB
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
#include "trueform/cpp/geometry/face_quality.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, std::enable_if_t<Dims == 3, int> = 0>
auto face_quality(Resolver &&resolver,
                  const cpp::mesh<Index, Real, Dims, Ngon> &value) {
  return submit<face_quality_result<Real>>(
      std::forward<Resolver>(resolver),
      [value] { return cpp::face_quality<Index, Real, Dims, Ngon>(value); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto face_quality(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<face_quality_result<Real>> {
  return async::face_quality<future_resolver, Index, Real, Dims, Ngon>(
      future_resolver{}, value);
}

} // namespace tf::cpp::async
