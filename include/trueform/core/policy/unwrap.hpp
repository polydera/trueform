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

namespace tf {
template <typename T> auto unwrap(T &&t) -> T && {
  return static_cast<T &&>(t);
}

template <typename T0, typename T1>
auto wrap_like(const T0 &, T1 &&t) -> T1 && {
  return static_cast<T1 &&>(t);
}

template <typename T> auto unwrapped(T &&t) -> decltype(auto) {
  auto &&unwrap_ = unwrap(static_cast<T &&>(t));
  if constexpr (std::is_same_v<std::decay_t<T>,
                               std::decay_t<decltype(unwrap_)>>)
    if constexpr (std::is_rvalue_reference_v<T &&>)
      return static_cast<T>(t);
    else
      return static_cast<T &&>(t);
  else
    return unwrapped(unwrap_);
}

template <typename T, typename F> auto wrap_map(T &&t, const F& map) -> decltype(auto) {
  auto &&unwrap_ = unwrap(static_cast<T &&>(t));
  if constexpr (std::is_same_v<std::decay_t<T>,
                               std::decay_t<decltype(unwrap_)>>)
    return map(static_cast<T&&>(t));
  else
    return wrap_like(t, wrap_map(unwrap_, map));
}

namespace policy {
// MSVC workaround: unqualified lookup from tf::policy should find tf::unwrap
// in the enclosing namespace, but MSVC fails when ADL also finds candidates
// (tf::core::unwrap). Explicitly importing the fallback fixes this.
using ::tf::unwrap;
using ::tf::wrap_like;

struct plain_op {};
template <typename T> auto operator|(T &&t, plain_op) -> decltype(auto) {
  return wrap_like(t, unwrapped(static_cast<T &&>(t)));
}
} // namespace policy
inline auto plain() -> policy::plain_op { return {}; }
} // namespace tf
