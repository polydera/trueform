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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/async/future_state.hpp"
#include "trueform/cpp/core/async/submit.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/fill/filled_mesh.hpp"
#include "trueform/cpp/fill/hole_fill_report.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {
template <typename Resolver, typename Index, typename Real>
auto filled_mesh(Resolver &&resolver, const mesh<Index, Real, 3> &value,
                 const hole_fill_report<Index, Real> &report)
    -> resolver_result_t<Resolver, tf::polygons_buffer<Index, Real, 3, 3>> {
  using result_type = tf::polygons_buffer<Index, Real, 3, 3>;
  return submit<result_type>(std::forward<Resolver>(resolver), [value, report] {
    return cpp::filled_mesh<Index, Real>(value, report);
  });
}

template <typename Index, typename Real>
auto filled_mesh(const mesh<Index, Real, 3> &value,
                 const hole_fill_report<Index, Real> &report)
    -> std::future<tf::polygons_buffer<Index, Real, 3, 3>> {
  return async::filled_mesh<future_resolver, Index, Real>(future_resolver{},
                                                          value, report);
}

} // namespace tf::cpp::async
