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
#include "trueform/cpp/core/async/completion.hpp"

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <tbb/task_group.h>

namespace tf::cpp::async {
namespace detail {

class runtime {
  std::mutex _state_mutex;
  std::condition_variable _drained;
  std::size_t _in_flight{0};
  bool _accepting{true};

  std::mutex _group_mutex;
  tbb::task_group _group;

  auto finish_task() noexcept -> void {
    {
      std::lock_guard<std::mutex> lock(_state_mutex);
      --_in_flight;
    }
    _drained.notify_all();
  }

public:
  std::mutex vendor_mutex;
  /// Held behind a handle, so a completion takes a share of it rather than a
  /// copy of the callable: a throwing copy inside a noexcept publication
  /// terminates, and every completion would pay for one.
  std::shared_ptr<const completion_vendor> vendor;

  ~runtime() noexcept {
    try {
      {
        std::lock_guard<std::mutex> lock(_state_mutex);
        _accepting = false;
      }
      drain();
      static_cast<void>(_group.wait());
    } catch (...) {
    }
  }

  auto enqueue(std::function<void()> function) -> void {
    {
      std::lock_guard<std::mutex> lock(_state_mutex);
      if (!_accepting)
        throw std::runtime_error("async executor is closed");
      ++_in_flight;
    }

    try {
      std::lock_guard<std::mutex> lock(_group_mutex);
      _group.run([this, function = std::move(function)]() noexcept {
        try {
          function();
        } catch (...) {
        }
        finish_task();
      });
    } catch (...) {
      finish_task();
      throw;
    }
  }

  auto drain() -> void {
    std::unique_lock<std::mutex> lock(_state_mutex);
    _drained.wait(lock, [this] { return _in_flight == 0; });
  }
};

auto common_runtime() -> runtime & {
  static runtime instance;
  return instance;
}

auto enqueue(std::function<void()> function) -> void {
  common_runtime().enqueue(std::move(function));
}

auto publish_completion(std::shared_ptr<completion_base> state) noexcept
    -> void {
  std::shared_ptr<const completion_vendor> vendor;
  {
    auto &instance = common_runtime();
    std::lock_guard<std::mutex> lock(instance.vendor_mutex);
    vendor = instance.vendor;
  }
  completion finished(std::move(state));
  if (!vendor) {
    finished.publish();
    return;
  }
  try {
    (*vendor)(std::move(finished));
  } catch (...) {
  }
}

} // namespace detail

auto set_completion_vendor(completion_vendor vendor) -> void {
  auto held = vendor ? std::make_shared<const completion_vendor>(
                           std::move(vendor))
                     : nullptr;
  auto &instance = detail::common_runtime();
  std::lock_guard<std::mutex> lock(instance.vendor_mutex);
  instance.vendor = std::move(held);
}

} // namespace tf::cpp::async
