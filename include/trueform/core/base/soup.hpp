/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>

namespace tf::core {

template <typename Policy> struct soup : Policy {
  soup(const Policy &r) : Policy{r} {}
  soup(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const soup<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(soup<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(soup<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const soup<Policy> &, T &&t) {
  return soup<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(soup<Policy> &, T &&t) {
  return soup<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(soup<Policy> &&, T &&t) {
  return soup<std::decay_t<T>>{static_cast<T &&>(t)};
}
} // namespace tf::core

