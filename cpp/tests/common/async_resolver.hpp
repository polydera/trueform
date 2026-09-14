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

#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf::cpp::test {

/// Resolver whose copies share an atomic submission counter.
class counting_resolver {
  std::shared_ptr<std::atomic<std::size_t>> _submissions;

public:
  counting_resolver()
      : _submissions(std::make_shared<std::atomic<std::size_t>>(0)) {}

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    _submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }

  auto submissions() const -> std::size_t {
    return _submissions->load(std::memory_order_relaxed);
  }

  auto reset() const -> void {
    _submissions->store(0, std::memory_order_relaxed);
  }
};

template <typename Invoke>
auto resolve_once(counting_resolver &resolver, Invoke &&invoke) {
  const auto before = resolver.submissions();
  auto pending = std::forward<Invoke>(invoke)(resolver);
  if (resolver.submissions() != before + 1)
    throw std::logic_error("async operation did not submit exactly once");

  using result_type = decltype(pending.get());
  if constexpr (std::is_void<result_type>::value) {
    pending.get();
  } else {
    return pending.get();
  }
}

} // namespace tf::cpp::test
