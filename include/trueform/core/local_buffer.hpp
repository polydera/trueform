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
#include "./buffer.hpp"
#include "./cache_aligned_slot.hpp"
#include "./views/mapped_range.hpp"
#include "tbb/task_arena.h"
#include <vector>

namespace tf {

/// @ingroup core_buffers
/// @brief A thread-local buffer container for use within a TBB task arena.
///
/// `local_buffer` provides a buffer-like interface where each thread in the
/// current TBB task arena transparently operates on its own private
/// `std::buffer<T>` instance, identified by
/// `tbb::this_task_arena::current_thread_index()`.
///
/// This container is designed for efficient parallel algorithms where each
/// thread performs local accumulation (e.g., via `push_back`, `emplace_back`,
/// etc.) without contention or locking. Once parallel work is done, the results
/// can be merged into a single `std::buffer<T>` using `.to_buffer()`.
///
/// ### Key Properties:
/// - Thread-safe by design (one buffer per thread, no locks)
/// - No synchronization needed within `push_back` or `emplace_back`
/// - Should only be used from threads managed by the TBB task scheduler
/// - Not copyable or movable
///
/// @tparam T The value type stored in the buffer.
template <typename T> class local_buffer {
public:
  local_buffer() : _buffers(tbb::this_task_arena::max_concurrency()) {}

  local_buffer(const local_buffer &) = delete;
  local_buffer(local_buffer &&) = delete;
  auto operator=(const local_buffer &) -> local_buffer & = delete;
  auto operator=(local_buffer &&) -> local_buffer & = delete;

  /// @brief Constructs an objects into the current thread's buffer.
  template <typename... Args> auto emplace_back(Args &&...args) -> T & {
    auto &vec = local();
    vec.emplace_back(static_cast<Args &&>(args)...);
    return vec.back();
  }

  /// @brief Pushes a value into the current thread's buffer.
  auto push_back(T value) -> T & {
    auto &vec = local();
    vec.push_back(std::move(value));
    return vec.back();
  }

  /// @brief Local size
  auto size() const -> std::size_t { return local().size(); }

  auto total_size() const -> std::size_t {
    std::size_t total = 0;
    for (const auto &v : _buffers)
      total += v.value.size();
    return total;
  }

  /// @brief Local capacity
  auto capacity() const -> std::size_t { return local().capacity(); }

  /// @brief Returns a merged buffer from all threads.

  auto to_buffer(tf::buffer<T> &out) const -> void {
    out.allocate(total_size());
    auto it = out.begin();
    for (const auto &v : _buffers)
      it = std::copy(v.value.begin(), v.value.end(), it);
  }

  auto to_buffer() const -> tf::buffer<T> {
    tf::buffer<T> out;
    to_buffer(out);
    return out;
  }

  template <typename Iterator>
  auto to_iterator(Iterator out) const -> Iterator {
    for (const auto &v : _buffers)
      out = std::copy(v.value.begin(), v.value.end(), out);
    return out;
  }

  /// @brief Clears only the local buffer.
  void clear() { local().clear(); }

  /// @brief Clearsall buffers
  void clear_all() {
    for (auto &v : _buffers)
      v.value.clear();
  }

  /// @brief Reserve capacity in the local buffer.
  void reserve(std::size_t n) { local().reserve(n); }

  void reserve_all(std::size_t n) {
    for (auto &b : _buffers)
      b.value.reserve(n);
  }

  /// @brief Resizes  the local buffer.
  void allocate(std::size_t n) { local().allocate(n); }

  void reallocate(std::size_t n) { local().reallocate(n); }

  /// @brief Access element in the local buffer.
  auto operator[](std::size_t i) -> T & { return local()[i]; }

  auto operator[](std::size_t i) const -> const T & { return local()[i]; }

  /// @brief Return iterator to beginning of local buffer.
  auto begin() -> typename tf::buffer<T>::iterator { return local().begin(); }

  auto begin() const -> typename tf::buffer<T>::const_iterator {
    return local().begin();
  }

  /// @brief Return iterator to end of local buffer.
  auto end() -> typename tf::buffer<T>::iterator { return local().end(); }

  auto end() const -> typename tf::buffer<T>::const_iterator {
    return local().end();
  }

  /// @brief Check if local buffer is empty.
  auto empty() const -> bool { return local().empty(); }

  auto operator*() const -> const tf::buffer<T> & { return local(); }

  auto operator*() -> tf::buffer<T> & { return local(); }

  auto buffers() const {
    return tf::make_mapped_range(
        _buffers, [](const auto &x) -> const tf::buffer<T> & { return x.value; });
  }

private:
  /// @brief Returns reference to the local buffer.
  auto local() -> tf::buffer<T> & {
    return _buffers[tbb::this_task_arena::current_thread_index()].value;
  }

  /// @brief Returns const reference to the local buffer.
  auto local() const -> const tf::buffer<T> & {
    return _buffers[tbb::this_task_arena::current_thread_index()].value;
  }

  std::vector<core::cache_aligned_slot<tf::buffer<T>>> _buffers;
};

} // namespace tf
