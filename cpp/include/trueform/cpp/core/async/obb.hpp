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
#include "trueform/cpp/core/obb.hpp"

#include <future>
#include <utility>

namespace tf::cpp::async {

template <typename Resolver, typename Real>
auto obb_from(Resolver &&resolver, const obb_options<Real> &options)
    -> resolver_result_t<Resolver, obb_result<Real>> {
  return submit<obb_result<Real>>(std::forward<Resolver>(resolver),
                                  [options] { return cpp::obb_from(options); });
}

template <typename Real>
auto obb_from(const obb_options<Real> &options)
    -> std::future<obb_result<Real>> {
  return async::obb_from(future_resolver{}, options);
}

} // namespace tf::cpp::async
