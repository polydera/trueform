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

#include "trueform/cpp/core/async/completion.hpp"
#include "trueform/cpp/core/async/future_state.hpp"

#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <utility>

namespace tf::cpp::async {
template <typename F>
using result_t = std::decay_t<std::invoke_result_t<std::decay_t<F> &>>;

template <typename Resolver, typename T>
using resolver_state_t =
    typename std::decay_t<Resolver>::template state_type<T>;

template <typename Resolver, typename T>
using resolver_result_t =
    decltype(std::declval<resolver_state_t<Resolver, T> &>().result());

/// @brief Resolve one callable on the common TBB executor.
///
/// THE LIFETIME CONTRACT, stated once for every async entry above this: what a
/// job carries is a mesh, an edge mesh or a point cloud AS IT STANDS — one
/// coherent reading of memory the CALLER keeps alive until the future
/// completes, which is the same borrow law a synchronous call obeys. A caller
/// whose storage cannot outlive the call assembles with the keepalive the
/// carrier's constructor takes; a binding captures its own shared handles that
/// way. Nothing here copies geometry on a caller's behalf.
///
/// AN ARRAY IS CARRIED THE SAME WAY: an `nd_array` or an
/// `offset_blocked_buffer` argument is captured as the HANDLE it is, which
/// shares the storage and retains whatever owner that storage was built with.
/// So a job reads the caller's values, and a caller that means the job to see
/// call-time values does not mutate them until the future completes — the one
/// law, for the carrier and for the arrays beside it. No entry snapshots an
/// array: a deep copy of a million scalars per dispatch is a cost the average
/// caller does not owe.
///
/// THE CACHE CONTRACT rides with it: a cache shared by concurrent jobs is
/// filled before it is shared — the `build_*` verbs are that ask — or
/// synchronized by the caller. One job filling on its own worker is one
/// filler, and needs no ceremony. What concurrent jobs share is the geometry
/// and the cache; each job carries a mesh VALUE of its own, which is what the
/// capture below makes of the one it was handed.
template <typename T, typename Resolver, typename F>
auto submit(Resolver &&resolver, F &&function)
    -> resolver_result_t<Resolver, T> {
  auto state = std::forward<Resolver>(resolver).template make_state<T>();
  static_assert(
      std::is_same<decltype(state),
                   std::shared_ptr<resolver_state_t<Resolver, T>>>::value,
      "resolver make_state<T>() must return shared_ptr<state_type<T>>");
  static_assert(
      noexcept(state->set_exception(std::declval<std::exception_ptr>())),
      "resolver state set_exception must be noexcept");
  auto result = state->result();

  try {
    auto shared_function =
        std::make_shared<std::decay_t<F>>(std::forward<F>(function));
    detail::enqueue([state, shared_function]() noexcept {
      try {
        if constexpr (std::is_void<T>::value) {
          std::invoke(*shared_function);
          state->set_value();
        } else {
          state->set_value(std::invoke(*shared_function));
        }
      } catch (...) {
        state->set_exception(std::current_exception());
      }
    });
  } catch (...) {
    state->set_exception(std::current_exception());
  }

  return result;
}

template <typename Resolver, typename F>
auto submit(Resolver &&resolver, F &&function)
    -> decltype(submit<result_t<F>>(std::forward<Resolver>(resolver),
                                    std::forward<F>(function))) {
  return submit<result_t<F>>(std::forward<Resolver>(resolver),
                             std::forward<F>(function));
}

template <typename F> auto submit(F &&function) -> std::future<result_t<F>> {
  return submit<result_t<F>>(future_resolver{}, std::forward<F>(function));
}

} // namespace tf::cpp::async
