/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>

namespace tf {
template <typename Policy> struct face_membership_like : Policy {
  face_membership_like() = default;
  face_membership_like(const Policy &policy) : Policy{policy} {}
  face_membership_like(Policy &&policy) : Policy{std::move(policy)} {}
};

template <typename Policy>
auto unwrap(const face_membership_like<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(face_membership_like<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(face_membership_like<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const face_membership_like<Policy> &, T &&t) {
  return face_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(face_membership_like<Policy> &, T &&t) {
  return face_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(face_membership_like<Policy> &&, T &&t) {
  return face_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_face_membership_like(Range &&r) {
  return tf::face_membership_like<std::decay_t<Range>>{
      static_cast<Range &&>(r)};
}
} // namespace tf
