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
#include "tbb/task_arena.h"
#include <vector>

namespace tf {

/// @ingroup core_algorithms
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
      out.insert(out.end(), v.value.begin(), v.value.end());
    return out;
  }

  template <typename Iterator>
  auto to_iterator(Iterator out) const -> Iterator {
    for (const auto &v : _vectors)
      out = std::copy(v.value.begin(), v.value.end(), out);
    return out;
  }

  /// @brief Clears only the local vector.
  void clear() { local().clear(); }

  /// @brief Clearsall vectors
  void clear_all() {
    for (auto &v : _vectors)
      v.value.clear();
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

  auto operator*() const -> const std::vector<T> & { return local(); }

  auto operator*() -> std::vector<T> & { return local(); }

  auto vectors() const {
    return tf::make_mapped_range(
        _vectors,
        [](const auto &x) -> const std::vector<T> & { return x.value; });
  }

private:
  /// @brief Returns reference to the local vector.
  auto local() -> std::vector<T> & {
    return _vectors[tbb::this_task_arena::current_thread_index()].value;
  }

  /// @brief Returns const reference to the local vector.
  auto local() const -> const std::vector<T> & {
    return _vectors[tbb::this_task_arena::current_thread_index()].value;
  }

  auto total_size() const -> std::size_t {
    std::size_t total = 0;
    for (const auto &v : _vectors)
      total += v.value.size();
    return total;
  }

  std::vector<core::cache_aligned_slot<std::vector<T>>> _vectors;
};

} // namespace tf
