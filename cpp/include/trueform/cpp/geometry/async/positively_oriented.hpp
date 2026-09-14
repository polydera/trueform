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
#include "trueform/cpp/geometry/positively_oriented.hpp"

#include <cstddef>
#include <future>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Dims,
          std::size_t Ngon, std::enable_if_t<Dims == 3, int> = 0>
auto positively_oriented(Resolver &&resolver,
                         const cpp::mesh<Index, Real, Dims, Ngon> &value,
                         bool is_consistent = false) {
  return submit<tf::polygons_buffer<Index, Real, Dims, Ngon>>(
      std::forward<Resolver>(resolver), [value, is_consistent] {
        return cpp::positively_oriented<Index, Real, Dims, Ngon>(value,
                                                                 is_consistent);
      });
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto positively_oriented(const cpp::mesh<Index, Real, Dims, Ngon> &value,
                         bool is_consistent = false)
    -> std::future<tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  return async::positively_oriented<future_resolver, Index, Real, Dims, Ngon>(
      future_resolver{}, value, is_consistent);
}

} // namespace tf::cpp::async
