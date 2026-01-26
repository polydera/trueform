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
#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

namespace tf {
/// @ingroup core_buffers
/// @brief A minimal, trivially-constructible alternative to `std::vector` for
/// POD types.
///
/// `tf::buffer<T>` is a lightweight dynamic array for trivially constructible
/// and trivially destructible types. It supports the same general interface as
/// `std::vector`, with one exception.
///
/// Unlike `std::vector`, `tf::buffer<T>::allocate(n)` and
/// `tf::buffer<T>::reallocate()` allocate memory for `n` elements **without
/// initializing them**.
///
/// Use `tf::buffer<T>` only for simple POD-like types. For anything with
/// constructors or destructors, use `std::vector<T>` instead.
///
/// @tparam T A trivially constructible and trivially destructible type.
///
/// @note This container is used internally.
/// @see `allocate()` and `reallocate()` for uninitialized memory allocation.
template <typename T> class buffer {
  static_assert(std::is_trivially_default_constructible<T>::value &&
                    std::is_trivially_destructible<T>::value,
                "Just use std::vector");

public:
  using value_type = T;
  using reference = T &;
  using pointer = T *;
  using iterator = T *;
  using const_iterator = const T *;
  using size_type = std::size_t;
  /// @brief Default constructor. Creates an empty buffer.
  buffer() = default;

  /// @brief Copy constructor. Performs a deep copy of the buffer contents.
  buffer(const buffer &other) {
    auto size = other.size();
    allocate(size);
    std::memcpy(_data.get(), other._data.get(), size * sizeof(T));
  }

  /// @brief Move constructor. Transfers ownership of buffer memory.
  buffer(buffer &&other) noexcept {
    _data = std::move(other._data);
    _end = other._end;
    _capacity = other._capacity;
    other._end = other._data.get();
    other._capacity = other._data.get();
  }

  /// @brief Copy assignment. Performs a deep copy of the buffer contents.
  auto operator=(const buffer &other) -> buffer & {
    auto size = other.size();
    allocate(size);
    std::memcpy(_data.get(), other._data.get(), size * sizeof(T));
    return *this;
  }

  /// @brief Move assignment. Transfers ownership of buffer memory.
  auto operator=(buffer &&other) noexcept -> buffer & {
    _data = std::move(other._data);
    _end = other._end;
    _capacity = other._capacity;
    other._end = other._data.get();
    other._capacity = other._data.get();
    return *this;
  }

  /// @brief Allocates uninitialized memory for `n` elements.
  auto allocate(std::size_t n) {
    if (n <= capacity()) {
      _end = _data.get() + n;
    } else {
      auto new_capacity = compute_new_capacity(n - size());
      std::unique_ptr<T[]> tmp{new T[new_capacity]};
      std::swap(tmp, _data);
      _end = _data.get() + n;
      _capacity = _data.get() + new_capacity;
    }
  }

  /// @brief Reallocates the buffer to size `n`, preserving current content.
  auto reallocate(std::size_t n) {
    if (n <= capacity()) {
      _end = _data.get() + n;
    } else
      append_at_end(n - size());
  }

  /// @brief Reallocates the buffer to size `n`, filling new elements with value
  /// `t`.
  auto reallocate_and_initialize(std::size_t n, T t) {
    if (n <= capacity()) {
      _end = _data.get() + n;
    } else {
      auto _size = size();
      append_at_end(n - _size);
      std::fill(_data.get() + _size, _end, std::move(t));
    }
  }

  /// @brief Allocates memory for `n` elements and initializes all with value
  /// `t`.
  auto allocate_and_initialize(std::size_t n, T t) {
    allocate(n);
    std::fill(_data.get(), _end, std::move(t));
  }

  /// @brief Erases elements from a given new end pointer to the actual end.
  auto erase_till_end(T *const _new_end) { _end = _new_end; }

  /// @brief Erases a range of elements and compacts the remaining content.
  auto erase(T *_from, const T *_to) {
    auto it = _to;
    while (it != _end) {
      *_from++ = *it++;
    }
    _end = _from;
    return _end;
  }

  /// @brief Appends an element to the buffer (copy version).
  auto push_back(const T &t) -> T & {
    append_at_end(1);
    *(_end - 1) = t;
    return *(_end - 1);
  }

  /// @brief Appends an element to the buffer (move version).
  auto push_back(T &&t) -> T & {
    append_at_end(1);
    *(_end - 1) = std::move(t);
    return *(_end - 1);
  }

  /// @brief Removes the last element from the buffer.
  auto pop_back() { --_end; }

  /// @brief Constructs a new element in-place at the end of the buffer.
  template <typename... Us> auto emplace_back(Us &&...us) -> T & {
    append_at_end(1);
    new (_end - 1) T{static_cast<Us &&>(us)...};
    return *(_end - 1);
  }

  /// @brief Clears the buffer but retains allocated memory.
  auto clear() { _end = _data.get(); }

  /// @brief Ensures capacity for at least `n` elements.
  auto reserve(std::size_t n) {
    auto _size = size();
    if (n > _size) {
      append_at_end(n - _size, n);
      _end = _data.get() + _size;
    }
  }

  /// @brief Destroys the buffer and releases all memory.
  auto reset() {
    _data.reset(nullptr);
    _end = _data.get();
    _capacity = _end;
  }

  /// @brief Releases ownership of the buffer memory and returns the raw
  /// pointer.
  auto release() {
    auto out = _data.release();
    _end = _data.get();
    _capacity = _end;
    return out;
  }

  /// @brief Const element access by index.
  auto operator[](std::size_t i) const -> const T & { return *(begin() + i); }

  /// @brief Mutable element access by index.
  auto operator[](std::size_t i) -> T & { return *(begin() + i); }

  /// @brief Returns the number of elements the buffer can hold.
  auto capacity() const -> std::size_t { return _capacity - _data.get(); }

  /// @brief Returns the number of elements currently stored.
  auto size() const -> std::size_t { return _end - _data.get(); }

  /// @brief Returns an iterator to the beginning of the buffer.
  auto begin() -> T * { return _data.get(); }
  ///
  /// @brief Returns an iterator to the end of the buffer.
  auto end() -> T * { return _end; }

  /// @brief Returns a const iterator to the beginning of the buffer.
  auto begin() const -> const T * { return _data.get(); }

  /// @brief Returns a const iterator to the end of the buffer.
  auto end() const -> const T * { return _end; }

  /// @brief Returns a const iterator to the beginning of the buffer.
  auto cbegin() const -> const T * { return _data.get(); }

  /// @brief Returns a const iterator to the end of the buffer.
  auto cend() const -> const T * { return _end; }

  /// @brief Returns a reference to the first element.
  auto front() -> T & { return *_data.get(); }

  /// @brief Returns a const reference to the first element.
  auto front() const -> const T & { return *_data.get(); }

  /// @brief Returns a reference to the last element.
  auto back() -> T & { return *(_end - 1); }

  /// @brief Returns a const reference to the last element.
  auto back() const -> const T & { return *(_end - 1); }

  /// @brief Returns a const iterator to the beginning of the buffer.
  auto data() const -> const T * { return _data.get(); }

  /// @brief Returns a iterator to the beginning of the buffer.
  auto data() -> T * { return _data.get(); }

private:
  auto append_at_end(std::size_t n, std::size_t new_capacity) {
    auto _old_size = size();
    std::unique_ptr<T[]> tmp{new T[new_capacity]};
    std::memcpy(tmp.get(), _data.get(), _old_size * sizeof(T));
    std::swap(tmp, _data);
    _end = _data.get() + _old_size + n;
    _capacity = _data.get() + new_capacity;
  }

  auto append_at_end(std::size_t n) {
    auto _old_size = size();
    if (capacity() - _old_size >= n) {
      _end = _end + n;
    } else {
      auto new_capacity = compute_new_capacity(n);
      append_at_end(n, new_capacity);
    }
  }

  auto compute_new_capacity(std::size_t added_elements) {
    return size() + std::max(size(), added_elements);
  }

  std::unique_ptr<T[]> _data = nullptr;
  T *_end = nullptr;
  T *_capacity = nullptr;
};
} // namespace tf
