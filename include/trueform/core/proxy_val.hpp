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
#include <utility>
namespace tf {
namespace core {

template <typename PtrLike> class const_proxy_val {

public:
  using reference = decltype(*std::declval<PtrLike>());

  const_proxy_val() = default;
  const_proxy_val(PtrLike _ptr) : _ptr{_ptr} {}
  const_proxy_val(const const_proxy_val &) = default;
  const_proxy_val(const_proxy_val &&) = default;
  auto operator=(const const_proxy_val &other) -> const_proxy_val & = delete;
  auto operator=(const_proxy_val &other) -> const_proxy_val & = delete;

  auto operator*() const -> decltype(auto) { return *_ptr; }

  auto operator*() -> decltype(auto) { return *_ptr; }

  auto operator->() const { return _ptr; }
  auto operator->() { return _ptr; }

private:
  PtrLike _ptr;
};

template <typename PtrLike> class proxy_val {
public:
  using reference = decltype(*std::declval<PtrLike>());

  proxy_val() = default;
  proxy_val(PtrLike _ptr) : _ptr{_ptr} {}
  proxy_val(const proxy_val &) = default;
  proxy_val(proxy_val &&) = default;
  auto operator=(const proxy_val &other) -> proxy_val & {
    *_ptr = *other;
    return *this;
  }

  auto operator=(proxy_val &&other) -> proxy_val & {
    *_ptr = *other;
    return *this;
  }

  template <typename T>
  auto operator=(const proxy_val<T> &other) -> std::enable_if_t<
      std::is_assignable_v<reference, typename proxy_val<T>::reference>,
      proxy_val &> {
    *_ptr = *other;
    return *this;
  }

  template <typename T>
  auto operator=(const const_proxy_val<T> &other) -> std::enable_if_t<
      std::is_assignable_v<reference, typename const_proxy_val<T>::reference>,
      proxy_val &> {
    *_ptr = *other;
    return *this;
  }

  auto operator*() const -> decltype(auto) { return *_ptr; }

  auto operator*() -> decltype(auto) { return *_ptr; }

  auto operator->() const { return _ptr; }
  auto operator->() { return _ptr; }

private:
  PtrLike _ptr;
};

} // namespace core
template <typename PtrLike>
class proxy_val
    : public std::conditional_t<
          std::is_assignable_v<decltype(*std::declval<PtrLike>()),
                               decltype(*std::declval<PtrLike>())>,
          core::proxy_val<PtrLike>, core::const_proxy_val<PtrLike>> {
  using base_t = std::conditional_t<
      std::is_assignable_v<decltype(*std::declval<PtrLike>()),
                           decltype(*std::declval<PtrLike>())>,
      core::proxy_val<PtrLike>, core::const_proxy_val<PtrLike>>;

public:
  using base_t::base_t;
  using base_t::operator=;
};
} // namespace tf
