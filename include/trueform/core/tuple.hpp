/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <tuple>
#include <type_traits>

namespace tf {
namespace core {
template <typename... Ts> struct tuple : std::tuple<Ts...> {
  using std::tuple<Ts...>::tuple;
  using std::tuple<Ts...>::operator=;
};

template <typename... Ts> struct const_tuple : std::tuple<Ts...> {
  using std::tuple<Ts...>::tuple;
  const_tuple(const const_tuple &) = default;
  const_tuple(const_tuple &&) = default;
  auto operator=(const const_tuple &) -> const_tuple & = delete;
  auto operator=(const_tuple &&) -> const_tuple & = delete;
};

template <typename... Ts>
using tuple_base = std::conditional_t<(... && std::is_assignable_v<Ts &, Ts>),
                                      tuple<Ts...>, const_tuple<Ts...>>;
} // namespace core

template <typename... Ts> struct tuple : core::tuple_base<Ts...> {
  using base_t = core::tuple_base<Ts...>;
  using base_t::base_t;
  using base_t::operator=;
  tuple(const tuple &) = default;
  tuple(tuple &&) = default;
  auto operator=(const tuple &) -> tuple & = default;
  auto operator=(tuple &&) -> tuple & = default;
};

template <typename... Ts> auto make_tuple(Ts &&...ts) {
  return tf::tuple<std::decay_t<Ts>...>{static_cast<Ts &&>(ts)...};
}

template <typename... Ts> auto forward_lref_as_tuple(Ts &&...ts) {
  return tf::tuple<std::conditional_t<std::is_rvalue_reference_v<Ts>,
                                      std::decay_t<Ts>, Ts>...>{
      static_cast<Ts &&>(ts)...};
}
} // namespace tf

namespace std {

template <std::size_t I, typename... Ts>
struct tuple_element<I, tf::tuple<Ts...>> : tuple_element<I, tuple<Ts...>> {};

template <typename... Ts>
struct tuple_size<tf::tuple<Ts...>> : tuple_size<tuple<Ts...>> {};

} // namespace std
