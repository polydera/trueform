/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <type_traits>

namespace tf {
template <typename T> auto unwrap(T &&t) -> T && {
  return static_cast<T &&>(t);
}

template <typename T0, typename T1>
auto wrap_like(const T0 &, T1 &&t) -> T1 && {
  return static_cast<T1 &&>(t);
}

template <typename T> auto unwrapped(T &&t) -> decltype(auto) {
  auto unwrap_ = unwrap(static_cast<T &&>(t));
  if constexpr (std::is_same_v<std::decay_t<T>,
                               std::decay_t<decltype(unwrap_)>>)
    return static_cast<T &&>(t);
  else
    return unwrapped(unwrap_);
}

namespace policy {
struct plain_op {};
template <typename T> auto operator|(T &&t, plain_op) -> decltype(auto) {
  return wrap_like(t, unwrapped(static_cast<T &&>(t)));
}
} // namespace policy
inline auto plain() -> policy::plain_op { return {}; }
} // namespace tf
