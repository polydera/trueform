/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <tuple>
#include <utility>

namespace tf {
namespace detail {
template <typename F, typename Tuple, std::size_t... Is>
auto apply(F &&f, std::index_sequence<Is...>, Tuple &&tuple) -> decltype(auto) {
  using std::get;
  return f(get<Is>(tuple)...);
}

} // namespace detail

template <typename F, typename Tuple>
auto apply(F &&f, Tuple &&tuple) -> decltype(auto) {
  constexpr std::size_t N = std::tuple_size_v<Tuple>;
  return detail::apply(static_cast<F &&>(f), std::make_index_sequence<N>{},
                       static_cast<Tuple &&>(tuple));
}
} // namespace tf
