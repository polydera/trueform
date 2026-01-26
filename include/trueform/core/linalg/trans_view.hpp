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
#include <type_traits>
namespace tf::linalg {
template <typename T, std::size_t Dims> struct trans_view {
  using element_type = T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;
  constexpr static std::size_t n_rows = Dims;
  constexpr static std::size_t n_columns = Dims + 1;

  trans_view() = default;
  trans_view(T *_trans_view) : _trans_view{_trans_view} {}
  trans_view(const trans_view &) = default;
  trans_view(trans_view &&) = default;

  auto operator=(const trans_view &other) -> trans_view & {
    for (std::size_t i = 0; i < Dims * (Dims + 1); ++i)
      _trans_view[i] = other._trans_view[i];
  }

  auto operator()(std::size_t i, std::size_t j) const -> decltype(auto) {
    return _trans_view[i * n_columns + j];
  }

  auto operator()(std::size_t i, std::size_t j) -> decltype(auto) {
    return _trans_view[i * n_columns + j];
  }

private:
  T *_trans_view;
};

template <typename T, std::size_t Dims> struct trans_view<const T, Dims> {
  using element_type = const T;
  using value_type = T;
  using coordinate_type = std::decay_t<T>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;
  constexpr static std::size_t n_rows = Dims;
  constexpr static std::size_t n_columns = Dims + 1;

  trans_view() = default;
  trans_view(const T *_trans_view) : _trans_view{_trans_view} {}
  trans_view(const trans_view &) = default;
  trans_view(trans_view &&) = default;

  auto operator=(const trans_view &other) -> trans_view & = delete;

  constexpr auto rows() const -> std::size_t { return Dims; }

  constexpr auto columns() const -> std::size_t { return Dims + 1; }

  auto operator()(std::size_t i, std::size_t j) const -> decltype(auto) {
    return _trans_view[i * n_columns + j];
  }

  auto operator()(std::size_t i, std::size_t j) -> decltype(auto) {
    return _trans_view[i * n_columns + j];
  }

private:
  const T *_trans_view;
};

} // namespace tf::linalg
