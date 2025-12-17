/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./mapped_range.hpp"
#include "./sequence_range.hpp"

namespace tf {
namespace views {
template <typename T> struct constant_policy {
  T value;
  template <typename U> auto operator()(U &&) const { return value; }
};
} // namespace views
template <typename T> auto make_constant_range(T &&value, std::size_t size) {
  return tf::make_mapped_range(
      tf::make_sequence_range(size),
      views::constant_policy<std::decay_t<T>>{static_cast<T &&>(value)});
}
} // namespace tf
