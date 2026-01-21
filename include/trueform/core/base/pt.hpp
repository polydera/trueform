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

#include "../zero.hpp"
#include <array>
#include <type_traits>
namespace tf::core {
template <typename T, std::size_t Size> struct pt;
template <typename T, std::size_t Dims> struct pt_view;

template <typename T, std::size_t Dims> struct pt_view {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  pt_view() = default;
  explicit pt_view(T *ptr) : _data(ptr) {}
  pt_view(const pt_view &other) : _data{other._data} {}
  pt_view(pt_view &&other) : _data{other._data} {}

  auto operator=(tf::zero_t) -> pt_view & {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = T(0);
    return *this;
  }

  auto operator=(const pt_view &other) -> pt_view & {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = other.data()[i];
    return *this;
  }

  template <typename U>
  auto operator=(const pt_view<U, Dims> &other)
      -> std::enable_if_t<std::is_assignable_v<T &, U>, pt_view &> {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = static_cast<T>(other.data()[i]);
    return *this;
  }

  template <typename U>
  auto operator=(const pt<U, Dims> &)
      -> std::enable_if_t<std::is_assignable_v<T &, U>, pt_view &>;

  auto data() -> T * { return _data; }
  auto data() const -> const T * { return _data; }

  friend auto swap(pt_view &&p0, pt_view &&p1) {
    for (std::size_t i = 0; i < Dims; ++i)
      std::swap(p0.data()[i], p1.data()[i]);
  }

private:
  T *_data;
};

template <typename T, std::size_t Dims> struct pt_view<const T, Dims> {
  using element_type = const T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  pt_view() = default;
  explicit pt_view(const T *ptr) : _data(ptr) {}
  pt_view(const pt_view &other) : _data{other._data} {}
  pt_view(pt_view &&other) : _data{other._data} {}

  auto operator=(const pt_view &other) -> pt_view & = delete;

  template <typename U>
  auto operator=(const pt_view<U, Dims> &other) -> pt_view & = delete;

  template <typename U>
  auto operator=(const pt<U, Dims> &) -> pt_view & = delete;

  auto data() -> const T * { return _data; }
  auto data() const -> const T * { return _data; }

private:
  const T *_data;
};

template <typename T, std::size_t Dims> struct pt {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  pt() = default;
  pt(tf::zero_t) : _data{} {}
  auto operator=(tf::zero_t) -> pt & {
    _data = {};
    return *this;
  }
  pt(std::array<T, Dims> _data) : _data{_data} {}
  template <typename... Ts,
            typename V = std::enable_if_t<
                (sizeof...(Ts) == Dims) &&
                    (... && std::is_convertible_v<std::decay_t<Ts>, T>),
                void>>
  pt(Ts &&...ts)
      : _data{static_cast<coordinate_type>(static_cast<Ts &&>(ts))...} {}

  template <typename U,
            typename V = std::enable_if_t<std::is_assignable_v<T &, U>, void>>
  pt(const pt_view<U, Dims> &other) {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = static_cast<T>(other.data()[i]);
  }

  template <typename U,
            typename V = std::enable_if_t<std::is_assignable_v<T &, U>, void>>
  pt(const pt<U, Dims> &other) {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = static_cast<T>(other.data()[i]);
  }

  template <typename U>
  auto operator=(const pt_view<U, Dims> &other)
      -> std::enable_if_t<std::is_assignable_v<T &, U>, pt &> {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = static_cast<T>(other.data()[i]);
    return *this;
  }

  template <typename U>
  auto operator=(const pt<U, Dims> &other)
      -> std::enable_if_t<std::is_assignable_v<T &, U>, pt &> {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = static_cast<T>(other.data()[i]);
    return *this;
  }

  auto data() -> T * { return _data.data(); }
  auto data() const -> const T * { return _data.data(); }

private:
  std::array<T, Dims> _data;
};

template <typename T, std::size_t Dims>
template <typename U>
auto pt_view<T, Dims>::operator=(const pt<U, Dims> &other)
    -> std::enable_if_t<std::is_assignable_v<T &, U>, pt_view &> {
  for (std::size_t i = 0; i < Dims; ++i)
    _data[i] = static_cast<T>(other.data()[i]);
  return *this;
}
} // namespace tf::core
