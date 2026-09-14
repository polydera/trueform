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

#include <functional>
#include <memory>
#include <utility>

namespace tf::cpp::async {
namespace detail {

auto enqueue(std::function<void()> function) -> void;

/// One finished operation whose outcome is stored and not yet published.
class completion_base {
public:
  virtual ~completion_base() = default;
  virtual auto publish() noexcept -> void = 0;
};

} // namespace detail

/// @brief A finished operation, its outcome held, awaiting publication.
///
/// The handle keeps the operation's state alive, so a vendor may carry it to
/// another thread and publish it there.
class completion {
  std::shared_ptr<detail::completion_base> _state;

public:
  explicit completion(std::shared_ptr<detail::completion_base> state)
      : _state(std::move(state)) {}

  /// Hand the stored outcome to whoever is waiting. Publishing more than once
  /// is a no-op, so a vendor never has to track whether it already did.
  auto publish() noexcept -> void {
    if (_state)
      _state->publish();
  }
};

/// @brief The runtime's one completion seam.
///
/// The callback runs on a worker thread the instant an operation's outcome is
/// stored. It belongs to whoever integrates this library into a language, and
/// is never visible to that language's users.
///
/// It must be cheap, non-blocking, and must never touch a language runtime: it
/// publishes the completion here, or it posts the completion to where a runtime
/// may publish it. Nothing else belongs in it.
///
/// The default vendor publishes on the spot, which is what makes the future
/// @ref submit returns ready as soon as the work is done.
using completion_vendor = std::function<void(completion)>;

auto set_completion_vendor(completion_vendor vendor) -> void;

namespace detail {

auto publish_completion(std::shared_ptr<completion_base> state) noexcept
    -> void;

} // namespace detail

} // namespace tf::cpp::async
