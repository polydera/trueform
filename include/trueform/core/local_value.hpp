/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "tbb/task_arena.h" // For tbb::this_task_arena
#include <vector>           // For std::vector

namespace tf {

/// @ingroup utilities
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
      : _values(tbb::this_task_arena::max_concurrency(), initial_value) {}

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
    T result = _values[0];
    for (std::size_t i = 1; i < _values.size(); ++i) {
      result = op(std::move(result), _values[i]);
    }
    return result;
  }

  /// @brief Resets all thread-local values to a given value.
  void reset(const T &reset_value) {
    for (auto &v : _values) {
      v = reset_value;
    }
  }

  auto values() -> std::vector<T> & { return _values; }

  auto values() const -> const std::vector<T> & { return _values; }

private:
  /// @brief Returns a reference to the current thread's local value.
  auto local() -> T & {
    return _values[tbb::this_task_arena::current_thread_index()];
  }

  /// @brief Returns a const reference to the current thread's local value.
  auto local() const -> const T & {
    return _values[tbb::this_task_arena::current_thread_index()];
  }

  std::vector<T> _values;
};

} // namespace tf
