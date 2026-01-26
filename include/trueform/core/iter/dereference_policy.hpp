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
#include <optional>

namespace tf::iter {
template <typename DereferncePolicy, bool = true>
struct dereference_policy_base : protected DereferncePolicy { // MSVC workaround
  dereference_policy_base() = default;
  dereference_policy_base(const DereferncePolicy &d) : DereferncePolicy{d} {}
  dereference_policy_base(DereferncePolicy &&d)
      : DereferncePolicy{std::move(d)} {}

  template <typename T> auto operator()(T &&t) const -> decltype(auto) {
    return static_cast<const DereferncePolicy &>(*this)(static_cast<T &&>(t));
  }

  template <typename T> auto operator()(T &&t) -> decltype(auto) {
    return static_cast<DereferncePolicy &>(*this)(static_cast<T &&>(t));
  }

  auto get() const -> const DereferncePolicy & {
    return static_cast<const DereferncePolicy &>(*this);
  }

  auto get() -> DereferncePolicy & {
    return static_cast<DereferncePolicy &>(*this);
  }
};

template <typename DereferncePolicy>
struct dereference_policy_base<DereferncePolicy, false> {
  std::optional<DereferncePolicy> policy;
  auto operator=(const dereference_policy_base &other)
      -> dereference_policy_base & {
    if (other.policy)
      policy.emplace(*(other.policy));
    else
      policy.reset();
    return *this;
  }

  auto operator=(dereference_policy_base &&other) -> dereference_policy_base & {
    if (other.policy)
      policy.emplace(std::move(*(other.policy)));
    else
      policy.reset();
    return *this;
  }

  dereference_policy_base(dereference_policy_base &&) = default;
  dereference_policy_base(const dereference_policy_base &) = default;
  dereference_policy_base() = default;
  dereference_policy_base(const DereferncePolicy &dp) : policy{dp} {}
  dereference_policy_base(DereferncePolicy &&dp) : policy{std::move(dp)} {}

  template <typename T> auto operator()(T &&t) const -> decltype(auto) {
    return (*policy)(static_cast<T &&>(t));
  }

  template <typename T> auto operator()(T &&t) -> decltype(auto) {
    return (*policy)(static_cast<T &&>(t));
  }

  auto get() const -> const DereferncePolicy & { return *policy; }

  auto get() -> DereferncePolicy & { return *policy; }
};

template <typename DereferncePolicy>
struct dereference_policy
    : dereference_policy_base<
          DereferncePolicy,
          std::is_default_constructible<DereferncePolicy>::value &&
              std::is_copy_assignable<DereferncePolicy>::value> {
  using base_t = dereference_policy_base<
      DereferncePolicy,
      std::is_default_constructible<DereferncePolicy>::value &&
          std::is_copy_assignable<DereferncePolicy>::value>;
  dereference_policy() = default;
  dereference_policy(const DereferncePolicy &d) : base_t{d} {}
  dereference_policy(DereferncePolicy &&d) : base_t{std::move(d)} {}
};
} // namespace tf::implementation
