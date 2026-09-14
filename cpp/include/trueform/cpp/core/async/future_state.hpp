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

#include <atomic>
#include <exception>
#include <future>
#include <memory>
#include <utility>

namespace tf::cpp::async {
namespace detail {

/// The default state: it stores the outcome, then hands itself to the vendor.
template <typename T>
class future_state final
    : public completion_base,
      public std::enable_shared_from_this<future_state<T>> {
  std::promise<T> _promise;
  // held indirectly: naming this state must not ask the result type to be
  // complete, since the typed entries are selected by substitution over it
  std::unique_ptr<T> _value;
  std::exception_ptr _exception;
  std::atomic<bool> _published{false};

public:
  future_state() = default;
  future_state(const future_state &) = delete;
  auto operator=(const future_state &) -> future_state & = delete;
  future_state(future_state &&) = delete;
  auto operator=(future_state &&) -> future_state & = delete;

  auto result() & -> std::future<T> { return _promise.get_future(); }

  template <typename U> auto set_value(U &&value) -> void {
    _value = std::make_unique<T>(std::forward<U>(value));
    publish_completion(this->shared_from_this());
  }

  auto set_exception(std::exception_ptr exception) noexcept -> void {
    _exception = std::move(exception);
    publish_completion(this->shared_from_this());
  }

  auto publish() noexcept -> void override {
    if (_published.exchange(true, std::memory_order_acq_rel))
      return;
    try {
      if (_exception)
        _promise.set_exception(std::move(_exception));
      else
        _promise.set_value(std::move(*_value));
    } catch (...) {
    }
  }
};

template <>
class future_state<void> final
    : public completion_base,
      public std::enable_shared_from_this<future_state<void>> {
  std::promise<void> _promise;
  std::exception_ptr _exception;
  std::atomic<bool> _published{false};

public:
  future_state() = default;
  future_state(const future_state &) = delete;
  auto operator=(const future_state &) -> future_state & = delete;
  future_state(future_state &&) = delete;
  auto operator=(future_state &&) -> future_state & = delete;

  auto result() & -> std::future<void> { return _promise.get_future(); }

  auto set_value() -> void { publish_completion(this->shared_from_this()); }

  auto set_exception(std::exception_ptr exception) noexcept -> void {
    _exception = std::move(exception);
    publish_completion(this->shared_from_this());
  }

  auto publish() noexcept -> void override {
    if (_published.exchange(true, std::memory_order_acq_rel))
      return;
    try {
      if (_exception)
        _promise.set_exception(std::move(_exception));
      else
        _promise.set_value();
    } catch (...) {
    }
  }
};

} // namespace detail

/// Stateless resolver producing the default native std::future result.
class future_resolver {
public:
  template <typename T> using state_type = detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    return std::make_shared<state_type<T>>();
  }
};

} // namespace tf::cpp::async
