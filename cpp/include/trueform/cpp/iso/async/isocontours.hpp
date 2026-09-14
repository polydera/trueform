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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/iso/isocontours.hpp"

#include <cstddef>
#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Index, typename Real, std::size_t Ngon>
auto isocontours(Resolver &&resolver, const mesh<Index, Real, 3, Ngon> &value,
                 const nd_array<Real> &scalars, Real cut_value)
    -> resolver_result_t<Resolver, tf::curves_buffer<Index, Real, 3>> {
  return submit<tf::curves_buffer<Index, Real, 3>>(
      std::forward<Resolver>(resolver), [value, scalars, cut_value] {
        return cpp::isocontours(value, scalars, cut_value);
      });
}

template <typename Index, typename Real, std::size_t Ngon>
auto isocontours(const mesh<Index, Real, 3, Ngon> &value,
                 const nd_array<Real> &scalars, Real cut_value)
    -> std::future<tf::curves_buffer<Index, Real, 3>> {
  return async::isocontours(future_resolver{}, value, scalars, cut_value);
}

template <typename Resolver, typename Index, typename Real, std::size_t Ngon>
auto isocontours(Resolver &&resolver, const mesh<Index, Real, 3, Ngon> &value,
                 const nd_array<Real> &scalars,
                 const nd_array<Real> &cut_values)
    -> resolver_result_t<Resolver, tf::curves_buffer<Index, Real, 3>> {
  return submit<tf::curves_buffer<Index, Real, 3>>(
      std::forward<Resolver>(resolver), [value, scalars, cut_values] {
        return cpp::isocontours(value, scalars, cut_values);
      });
}

template <typename Index, typename Real, std::size_t Ngon>
auto isocontours(const mesh<Index, Real, 3, Ngon> &value,
                 const nd_array<Real> &scalars,
                 const nd_array<Real> &cut_values)
    -> std::future<tf::curves_buffer<Index, Real, 3>> {
  return async::isocontours(future_resolver{}, value, scalars, cut_values);
}

} // namespace tf::cpp::async
