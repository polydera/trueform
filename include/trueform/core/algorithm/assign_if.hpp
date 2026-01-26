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
#include <atomic>
namespace tf {

/// @ingroup core_algorithms
/// @brief Conditionally assign to an atomic variable.
///
/// Atomically updates the variable if the compare predicate returns true.
/// Uses compare-exchange loop for thread-safe conditional update.
///
/// @tparam T The atomic value type.
/// @tparam F Binary predicate type.
/// @param atomic_var The atomic variable to update.
/// @param new_value The value to assign if compare succeeds.
/// @param compare Predicate: returns true if new_value should replace current.
/// @param initial_load Memory order for initial load.
/// @param publish_success Memory order on successful exchange.
/// @return True if the value was updated.
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
