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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/fill/fill_holes.hpp"
#include "trueform/cpp/fill/hole_fill_report.hpp"
#include "trueform/fill/hole_fill_config.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index, typename Real>
auto fill_holes(Resolver &&resolver, const mesh<Index, Real, 3> &value,
                tf::hole_fill_config config = {})
    -> resolver_result_t<Resolver, hole_fill_report<Index, Real>> {
  using result_type = hole_fill_report<Index, Real>;
  return submit<result_type>(std::forward<Resolver>(resolver), [value, config] {
    return cpp::fill_holes<Index, Real>(value, config);
  });
}

template <typename Index, typename Real>
auto fill_holes(const mesh<Index, Real, 3> &value,
                tf::hole_fill_config config = {})
    -> std::future<hole_fill_report<Index, Real>> {
  return async::fill_holes<future_resolver, Index, Real>(future_resolver{},
                                                         value, config);
}

template <typename Resolver, typename Index, typename Real>
auto fill_holes(Resolver &&resolver, const mesh<Index, Real, 3> &value,
                const nd_array<Index> &rim_ids,
                tf::hole_fill_config config = {})
    -> resolver_result_t<Resolver, hole_fill_report<Index, Real>> {
  using result_type = hole_fill_report<Index, Real>;
  return submit<result_type>(
      std::forward<Resolver>(resolver), [value, rim_ids, config] {
        return cpp::fill_holes<Index, Real>(value, rim_ids, config);
      });
}

template <typename Index, typename Real>
auto fill_holes(const mesh<Index, Real, 3> &value,
                const nd_array<Index> &rim_ids,
                tf::hole_fill_config config = {})
    -> std::future<hole_fill_report<Index, Real>> {
  return async::fill_holes<future_resolver, Index, Real>(
      future_resolver{}, value, rim_ids, config);
}

} // namespace tf::cpp::async
