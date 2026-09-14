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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/topology/orient_faces_consistently.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon>
auto orient_faces_consistently(Resolver &&resolver,
                               const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> resolver_result_t<Resolver,
                         tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  return submit<tf::polygons_buffer<Index, Real, Dims, Ngon>>(
      std::forward<Resolver>(resolver),
      [value] { return cpp::orient_faces_consistently(value); });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto orient_faces_consistently(const cpp::mesh<Index, Real, Dims, Ngon> &value)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  return async::orient_faces_consistently(future_resolver{}, value);
}

} // namespace tf::cpp::async
