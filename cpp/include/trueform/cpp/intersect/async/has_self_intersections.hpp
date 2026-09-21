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
#include "trueform/cpp/intersect/has_self_intersections.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Ngon>
auto has_self_intersections(Resolver &&resolver,
                            const mesh<Index, Real, 3, Ngon> &value)
    -> resolver_result_t<Resolver, bool> {
  return submit<bool>(std::forward<Resolver>(resolver),
                      [value] { return cpp::has_self_intersections(value); });
}

template <typename Index, typename Real, std::size_t Ngon>
auto has_self_intersections(const mesh<Index, Real, 3, Ngon> &value)
    -> std::future<bool> {
  return async::has_self_intersections(future_resolver{}, value);
}

} // namespace tf::cpp::async
