/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */

#pragma once

#include <array>
namespace tf::core {
template <typename T, std::size_t Dims> struct vec;
template <typename T, std::size_t Dims> struct vec_view;

template <typename T, std::size_t Dims> struct vec_view {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;

  vec_view() = default;
  explicit vec_view(T *vecr) : _data(vecr) {}
  vec_view(const vec_view &other) : _data{other._data} {}
  vec_view(vec_view &&other) : _data{other._data} {}

  auto operator=(const vec_view &other) -> vec_view & {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = other.data()[i];
    return *this;
  }

  template <typename U>
  auto operator=(const vec_view<U, Dims> &other)
      -> std::enable_if_t<std::is_assignable_v<T &, U>, vec_view &> {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = other.data()[i];
    return *this;
  }

  template <typename U>
  auto operator=(const vec<U, Dims> &)
      -> std::enable_if_t<std::is_assignable_v<T &, U>, vec_view &>;

  auto data() -> T * { return _data; }
  auto data() const -> const T * { return _data; }

private:
  T *_data;
};

template <typename T, std::size_t Dims> struct vec_view<const T, Dims> {
  using element_type = const T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;

  vec_view() = default;
  explicit vec_view(const T *vecr) : _data(vecr) {}
  vec_view(const vec_view &other) : _data{other._data} {}
  vec_view(vec_view &&other) : _data{other._data} {}

  auto operator=(const vec_view &other) -> vec_view & = delete;

  template <typename U>
  auto operator=(const vec_view<U, Dims> &other) -> vec_view & = delete;

  template <typename U>
  auto operator=(const vec<U, Dims> &) -> vec_view & = delete;

  auto data() -> T * { return _data; }
  auto data() const -> const T * { return _data; }

private:
  const T *_data;
};

template <typename T, std::size_t Dims> struct vec {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;

  vec() = default;
  vec(std::array<T, Dims> _data) : _data{_data} {}

  template <typename... Ts,
            typename V = std::enable_if_t<
                (sizeof...(Ts) == Dims) &&
                    (... && std::is_same_v<std::decay_t<Ts>, coordinate_type>),
                void>>
  vec(Ts &&...ts) : _data{static_cast<Ts &&>(ts)...} {}

  template <typename U,
            typename V = std::enable_if_t<std::is_assignable_v<T &, U>, void>>
  vec(const vec_view<U, Dims> &other) {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = other.data()[i];
  }

  template <typename U,
            typename V = std::enable_if_t<std::is_assignable_v<T &, U>, void>>
  vec(const vec<U, Dims> &other) {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = other.data()[i];
  }

  template <typename U>
  auto operator=(const vec_view<U, Dims> &other)
      -> std::enable_if_t<std::is_assignable_v<T &, U>, vec &> {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = other.data()[i];
    return *this;
  }

  template <typename U>
  auto operator=(const vec<U, Dims> &other)
      -> std::enable_if_t<std::is_assignable_v<T &, U>, vec &> {
    for (std::size_t i = 0; i < Dims; ++i)
      _data[i] = other.data()[i];
    return *this;
  }

  auto data() -> T * { return _data.data(); }
  auto data() const -> const T * { return _data.data(); }

private:
  std::array<T, Dims> _data;
};

template <typename T, std::size_t Dims>
template <typename U>
auto vec_view<T, Dims>::operator=(const vec<U, Dims> &other)
    -> std::enable_if_t<std::is_assignable_v<T &, U>, vec_view &> {
  for (std::size_t i = 0; i < Dims; ++i)
    _data[i] = other.data()[i];
  return *this;
}
} // namespace tf::core
