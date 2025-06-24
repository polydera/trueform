/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <atomic>
namespace tf {
template <typename T, typename F>
auto assign_if(std::atomic<T> &atomic_var, T new_value, const F &compare,
               std::memory_order initial_load = std::memory_order_relaxed,
               std::memory_order publish_success = std::memory_order_release)
    -> bool {
  T current = atomic_var.load(initial_load);
  while (compare(new_value, current)) {
    if (atomic_var.compare_exchange_weak(current, new_value, publish_success,
                                         std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}
} // namespace tf
