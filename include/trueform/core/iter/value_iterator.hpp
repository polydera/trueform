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
#include <iterator>
#include <type_traits>

namespace tf::iter {
template <typename T> class value_iterator {
  static_assert(std::is_integral_v<T>,
                "value_iterator<T>: T must be an integral type");

public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = T;

private:
  T _current;

public:
  explicit value_iterator(T value = 0) : _current(value) {}

  // Dereference
  auto operator*() const -> value_type { return _current; }

  // Pre-increment
  auto operator++() -> value_iterator & {
    ++_current;
    return *this;
  }

  // Post-increment
  auto operator++(int) -> value_iterator {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }

  // Pre-decrement
  auto operator--() -> value_iterator & {
    --_current;
    return *this;
  }

  // Post-decrement
  auto operator--(int) -> value_iterator {
    auto tmp = *this;
    --(*this);
    return tmp;
  }

  // Compound assignment
  auto operator+=(difference_type n) -> value_iterator & {
    _current += static_cast<T>(n);
    return *this;
  }

  auto operator-=(difference_type n) -> value_iterator & {
    _current -= static_cast<T>(n);
    return *this;
  }

  // Arithmetic
  auto operator+(difference_type n) const -> value_iterator {
    return value_iterator(_current + static_cast<T>(n));
  }

  auto operator-(difference_type n) const -> value_iterator {
    return value_iterator(_current - static_cast<T>(n));
  }

  auto operator[](difference_type n) const -> value_type {
    return _current + static_cast<T>(n);
  }

  // Distance
  friend auto operator-(const value_iterator &a, const value_iterator &b)
      -> difference_type {
    return static_cast<difference_type>(a._current - b._current);
  }

  // Comparison
  friend auto operator==(const value_iterator &a, const value_iterator &b)
      -> bool {
    return a._current == b._current;
  }

  friend auto operator!=(const value_iterator &a, const value_iterator &b)
      -> bool {
    return a._current != b._current;
  }

  friend auto operator<(const value_iterator &a, const value_iterator &b)
      -> bool {
    return a._current < b._current;
  }

  friend auto operator<=(const value_iterator &a, const value_iterator &b)
      -> bool {
    return a._current <= b._current;
  }

  friend auto operator>(const value_iterator &a, const value_iterator &b)
      -> bool {
    return a._current > b._current;
  }

  friend auto operator>=(const value_iterator &a, const value_iterator &b)
      -> bool {
    return a._current >= b._current;
  }
};

template <typename T> auto make_value(T t) { return value_iterator<T>{t}; }
} // namespace tf::iter
