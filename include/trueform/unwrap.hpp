/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
template <typename T> auto unwrap(T &&t) -> T && {
  return static_cast<T &&>(t);
}

template <typename T0, typename T1>
auto wrap_like(const T0 &, T1 &&t) -> T1 && {
  return static_cast<T1 &&>(t);
}
} // namespace tf
