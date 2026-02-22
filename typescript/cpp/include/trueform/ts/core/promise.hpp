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

#include "trueform/ts/core/async_dispatcher.hpp"
#include <utility>

namespace tf {
namespace ts {

/// @brief Return type alias for async dispatch slots.
using promise_t = uintptr_t;

/// @brief Dispatch a callable to TBB and return a slot address for JS
/// Atomics.waitAsync. Equivalent to `dispatcher.dispatch(fn)`.
///
/// Usage:
///   auto slot = tf::ts::promise([&]() { return do_work(); });
template <typename F> auto promise(F &&fn) -> promise_t {
  return dispatcher.dispatch(std::forward<F>(fn));
}

} // namespace ts
} // namespace tf
