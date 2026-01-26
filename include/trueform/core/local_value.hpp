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
#include "./cache_aligned_slot.hpp"
#include "./views/mapped_range.hpp"
#include "tbb/task_arena.h" // For tbb::this_task_arena
#include <vector>           // For std::vector

namespace tf {

/// @ingroup core_algorithms
/// @brief A thread-local value container for use within a TBB task arena.
///
/// `local_value` provides a value-like interface where each thread in the
/// current TBB task arena transparently operates on its own private
/// instance of type T, identified by
/// `tbb::this_task_arena::current_thread_index()`.
///
/// This container is designed for efficient parallel algorithms where each
/// thread performs local updates to a value (e.g., finding a thread-local
/// minimum, maximum, or sum) without contention or locking. Once parallel
/// work is complete, the results can be combined into a single value using
/// the `aggregate` method.
///
/// ### Key Properties:
/// - Thread-safe by design (one value per thread, no locks)
/// - No synchronization needed for access or modification
/// - Should only be used from threads managed by the TBB task scheduler
/// - Not copyable or movable
///
/// @tparam T The type of the thread-local value.
template <typename T> class local_value {
public:
  /// @brief Default constructor.
  /// Initializes a thread-local value for each potential thread.
  /// Requires T to be default-constructible.
  local_value() : _values(tbb::this_task_arena::max_concurrency()) {}

  /// @brief Constructor with an initial value.
  /// Initializes each thread-local value with a copy of `initial_value`.
  explicit local_value(const T &initial_value)
      : _values(tbb::this_task_arena::max_concurrency(),
                core::cache_aligned_slot<T>{initial_value}) {}

  local_value(const local_value &) = delete;
  local_value(local_value &&) = delete;
  auto operator=(const local_value &) -> local_value & = delete;
  auto operator=(local_value &&) -> local_value & = delete;

  /// @brief Dereference operator to access the thread-local value.
  auto operator*() -> T & { return local(); }

  /// @brief Const dereference operator.
  auto operator*() const -> const T & { return local(); }

  /// @brief Arrow operator to access members of the thread-local value.
  auto operator->() -> T * { return &local(); }

  /// @brief Const arrow operator.
  auto operator->() const -> const T * { return &local(); }

  /// @brief Aggregates all thread-local values into a single result.
  ///
  /// Uses the provided binary operation to combine the values. The aggregation
  /// starts with the value from the first thread and iteratively applies the
  /// operation with values from subsequent threads.
  ///
  /// @param op A binary function `(T, T) -> T` for aggregation (e.g.,
  /// std::plus, std::min).
  /// @return The final aggregated value.
  template <typename BinaryOp> auto aggregate(BinaryOp op) const -> T {
    // Start aggregation with the first thread's value.
    T result = _values[0].value;
    for (std::size_t i = 1; i < _values.size(); ++i) {
      result = op(std::move(result), _values[i].value);
    }
    return result;
  }

  /// @brief Resets all thread-local values to a given value.
  void reset(const T &reset_value) {
    for (auto &v : _values) {
      v.value = reset_value;
    }
  }

  auto values() {
    return tf::make_mapped_range(_values,
                                 [](auto &x) -> T & { return x.value; });
  }

  auto values() const {
    return tf::make_mapped_range(
        _values, [](const auto &x) -> const T & { return x.value; });
  }

private:
  auto get_id() const {
    return tbb::this_task_arena::current_thread_index();
    /*struct cache_t {*/
    /*  decltype(tbb::this_task_arena::current_thread_index()) _id;*/
    /*  const local_value<T> *_this;*/
    /*};*/
    /*static thread_local cache_t cache{*/
    /*    tbb::this_task_arena::current_thread_index(), this};*/
    /*if (cache._this != this)*/
    /*  cache = {tbb::this_task_arena::current_thread_index(), this};*/
    /*return cache._id;*/
  }
  /// @brief Returns a reference to the current thread's local value.
  auto local() -> T & { return _values[get_id()].value; }

  /// @brief Returns a const reference to the current thread's local value.
  auto local() const -> const T & { return _values[get_id()].value; }

  std::vector<core::cache_aligned_slot<T>> _values;
};

} // namespace tf
