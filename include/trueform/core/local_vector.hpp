/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "tbb/task_arena.h"
#include <vector>

namespace tf {

/// @ingroup utilities
/// @brief A thread-local vector container for use within a TBB task arena.
///
/// `local_vector` provides a vector-like interface where each thread in the
/// current TBB task arena transparently operates on its own private
/// `std::vector<T>` instance, identified by
/// `tbb::this_task_arena::current_thread_index()`.
///
/// This container is designed for efficient parallel algorithms where each
/// thread performs local accumulation (e.g., via `push_back`, `emplace_back`,
/// etc.) without contention or locking. Once parallel work is done, the results
/// can be merged into a single `std::vector<T>` using `.to_vector()`.
///
/// ### Key Properties:
/// - Thread-safe by design (one buffer per thread, no locks)
/// - No synchronization needed within `push_back` or `emplace_back`
/// - Should only be used from threads managed by the TBB task scheduler
/// - Not copyable or movable
///
/// @tparam T The value type stored in the vector.
template <typename T> class local_vector {
public:
  local_vector() : _vectors(tbb::this_task_arena::max_concurrency()) {}

  local_vector(const local_vector &) = delete;
  local_vector(local_vector &&) = delete;
  auto operator=(const local_vector &) -> local_vector & = delete;
  auto operator=(local_vector &&) -> local_vector & = delete;

  /// @brief Constructs an objects into the current thread's vector.
  template <typename... Args> auto emplace_back(Args &&...args) -> T & {
    auto &vec = local();
    vec.emplace_back(static_cast<Args &&>(args)...);
    return vec.back();
  }

  /// @brief Pushes a value into the current thread's vector.
  auto push_back(T value) -> T & {
    auto &vec = local();
    vec.push_back(std::move(value));
    return vec.back();
  }

  /// @brief Local size
  auto size() const -> std::size_t { return local().size(); }

  /// @brief Local capacity
  auto capacity() const -> std::size_t { return local().capacity(); }

  /// @brief Returns a merged vector from all threads.
  auto to_vector() const -> std::vector<T> {
    std::vector<T> out;
    out.reserve(total_size());
    for (const auto &v : _vectors)
      out.insert(out.end(), v.begin(), v.end());
    return out;
  }

  template<typename Iterator>
  auto to_iterator(Iterator out) const -> Iterator {
    for (const auto &v : _vectors)
      out = std::copy(v.begin(), v.end(), out);
    return out;
  }

  /// @brief Clears only the local vector.
  void clear() { local().clear(); }

  /// @brief Clearsall vectors
  void clear_all() {
    for (auto &v : _vectors)
      v.clear();
  }

  /// @brief Reserve capacity in the local vector.
  void reserve(std::size_t n) { local().reserve(n); }

  /// @brief Resizes  the local vector.
  void resize(std::size_t n) { local().resize(n); }

  /// @brief Access element in the local vector.
  auto operator[](std::size_t i) -> T & { return local()[i]; }

  auto operator[](std::size_t i) const -> const T & { return local()[i]; }

  /// @brief Return iterator to beginning of local vector.
  auto begin() -> typename std::vector<T>::iterator { return local().begin(); }

  auto begin() const -> typename std::vector<T>::const_iterator {
    return local().begin();
  }

  /// @brief Return iterator to end of local vector.
  auto end() -> typename std::vector<T>::iterator { return local().end(); }

  auto end() const -> typename std::vector<T>::const_iterator {
    return local().end();
  }

  /// @brief Check if local vector is empty.
  auto empty() const -> bool { return local().empty(); }

  auto vectors() const -> const std::vector<std::vector<T>> & {
    return _vectors;
  }

private:
  /// @brief Returns reference to the local vector.
  auto local() -> std::vector<T> & {
    return _vectors[tbb::this_task_arena::current_thread_index()];
  }

  /// @brief Returns const reference to the local vector.
  auto local() const -> const std::vector<T> & {
    return _vectors[tbb::this_task_arena::current_thread_index()];
  }

  auto total_size() const -> std::size_t {
    std::size_t total = 0;
    for (const auto &v : _vectors)
      total += v.size();
    return total;
  }

  std::vector<std::vector<T>> _vectors;
};

} // namespace tf
