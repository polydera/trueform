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

#include "trueform/core/curves_buffer.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/intersect/self_intersection_curves.hpp"
#include "trueform/intersect/intersect_config.hpp"
#include "trueform/intersect/intersect_mode.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Ngon>
auto self_intersection_curves(
    Resolver &&resolver, const mesh<Index, Real, 3, Ngon> &value,
    tf::intersect_config config = {tf::intersect_mode::sos |
                                   tf::intersect_mode::resolve_contours})
    -> resolver_result_t<Resolver, tf::curves_buffer<Index, Real, 3>> {
  return submit<tf::curves_buffer<Index, Real, 3>>(
      std::forward<Resolver>(resolver),
      [value, config] { return cpp::self_intersection_curves(value, config); });
}

template <typename Index, typename Real, std::size_t Ngon>
auto self_intersection_curves(
    const mesh<Index, Real, 3, Ngon> &value,
    tf::intersect_config config = {tf::intersect_mode::sos |
                                   tf::intersect_mode::resolve_contours})
    -> std::future<tf::curves_buffer<Index, Real, 3>> {
  return async::self_intersection_curves(future_resolver{}, value, config);
}

} // namespace tf::cpp::async
