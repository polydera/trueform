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
#include "../proxy_val.hpp"
#include "../static_size.hpp"
#include "../tuple.hpp"
#include "./type.hpp"
#include "./unwrap.hpp"
#include <type_traits>
#include <utility>

namespace tf {

namespace policy {
template <typename Index, typename Base> struct tag_state;
template <typename Index, typename Base> struct tag_state_ptr;

template <typename Range, typename Base>
auto has_state(type, const tag_state<Range, Base> *) -> std::true_type;
template <typename Range, typename Base>
auto has_state(type, const tag_state_ptr<Range, Base> *) -> std::true_type;

auto has_state(type, const void *) -> std::false_type;
} // namespace policy

template <typename T>
inline constexpr bool has_state_policy = decltype(has_state(
    policy::type{}, static_cast<const std::decay_t<T> *>(nullptr)))::value;
namespace policy {
/**
 * @ingroup core_policies
 * @brief Type injector that tags state to a class.
 *
 * Provides `.state()` accessors for arbitrary user-defined state.
 *
 * @tparam State The state type.
 * @tparam Base The type being augmented.
 */
template <typename State, typename Base> struct tag_state : Base {

  using Base::operator=;
  /**
   * @brief Constructs an instance.
   */
  tag_state(const State &_state, const Base &base)
      : Base{base}, _state{_state} {}

  /**
   * @brief Constructs an instance.
   */
  tag_state(State &&_state, Base &&base)
      : Base{std::move(base)}, _state{std::move(_state)} {}

  template <typename Other>
  auto operator=(Other &&other) -> std::enable_if_t<
      has_state_policy<Other> &&
          std::is_assignable_v<State &, decltype(other.state())> &&
          std::is_assignable_v<Base &, Other &&>,
      tag_state &> {
    Base::operator=(static_cast<Other &&>(other));
    state() = other.state();
    return *this;
  }

  /**
   * @brief Returns a const reference to the state.
   */
  auto state() const -> const State & { return _state; }

  /**
   * @brief Returns a mutable reference to the state.
   */
  auto state() -> State & { return _state; }

private:
  State _state;

  friend auto unwrap(const tag_state &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_state &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_state &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_state &val, T &&t) {
    return tag_state<State, std::decay_t<T>>{val.state(), static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_state<Range, Base>> : static_size<Base> {};

namespace policy {

template <typename Proxy, typename Base> struct tag_state_ptr : Base {

  using Base::operator=;
  /**
   * @brief Constructs an instance.
   */
  tag_state_ptr(const tf::proxy_val<Proxy> &_state, const Base &base)
      : Base{base}, _state{_state} {}

  /**
   * @brief Constructs an instance.
   */
  tag_state_ptr(tf::proxy_val<Proxy> &&_state, Base &&base)
      : Base{std::move(base)}, _state{std::move(_state)} {}

  template <typename Other>
  auto operator=(Other &&other) -> std::enable_if_t<
      has_state_policy<Other> &&
          std::is_assignable_v<decltype(*std::declval<Proxy>()),
                               decltype(other.state())> &&
          std::is_assignable_v<Base &, Other &&>,
      tag_state_ptr &> {
    Base::operator=(static_cast<Other &&>(other));
    state() = other.state();
    return *this;
  }

  /**
   * @brief Returns a const reference to the state.
   */
  auto state() const -> decltype(auto) { return *_state; }

  /**
   * @brief Returns a mutable reference to the state.
   */
  auto state() -> decltype(auto) { return *_state; }

private:
  tf::proxy_val<Proxy> _state;

  friend auto unwrap(const tag_state_ptr &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_state_ptr &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_state_ptr &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_state_ptr &val, T &&t) {
    return tag_state_ptr<Proxy, std::decay_t<T>>{val._state,
                                                 static_cast<T &&>(t)};
  }
};
} // namespace policy
template <typename Range, typename Base>
struct static_size<policy::tag_state_ptr<Range, Base>> : static_size<Base> {};

namespace policy {
template <typename Iterator, typename Base>
auto tag_state_ptr_impl(Iterator &&state, Base &&base) {
  if constexpr (has_state_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(base,
                     policy::tag_state_ptr<std::decay_t<Iterator>,
                                           std::decay_t<decltype(b_base)>>{
                         static_cast<Iterator &&>(state), b_base});
  }
}

template <typename Index, typename Base>
auto tag_state_impl(Index &&state, Base &&base) {
  if constexpr (has_state_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_state<std::decay_t<Index>, std::decay_t<decltype(b_base)>>{
            static_cast<Index &&>(state), b_base});
  }
}

struct state_ptr_tagger {
  template <typename Iter0, typename Iter1>
  auto operator()(std::pair<Iter0, Iter1> iters) const {
    return tag_state_ptr_impl(iters.first, *iters.second);
  }
};

template <typename T> struct tag_state_op {
  T state;
};

template <typename U, typename T> auto operator|(U &&u, tag_state_op<T> t) {
  return tag_state_impl(t.state, static_cast<U &&>(u));
}
} // namespace policy

template <typename Index> auto tag_state(Index &&state) {
  return policy::tag_state_op<std::decay_t<Index>>{
      static_cast<Index &&>(state)};
}
template <typename T, typename... Ts> auto tag_state(T &&t, Ts &&...states) {
  auto tup =
      tf::make_tuple(static_cast<T &&>(t), static_cast<Ts &&>(states)...);
  return policy::tag_state_op<decltype(tup)>{std::move(tup)};
}
} // namespace tf
namespace std {
template <typename Index, typename Policy>
struct tuple_size<tf::policy::tag_state<Index, Policy>> : tuple_size<Policy> {};

template <std::size_t I, typename Index, typename Policy>
struct tuple_element<I, tf::policy::tag_state<Index, Policy>> {
  using type = typename std::iterator_traits<
      decltype(declval<Policy>().begin())>::value_type;
};
} // namespace std
