/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./apply.hpp"
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {
namespace detail {

// Fetches the I-th element from each tuple and makes a group (e.g., std::tuple
// or std::pair)
template <std::size_t I, typename... Tuples>
auto make_group_at(Tuples &&...tuples) {
  return std::forward_as_tuple(std::get<I>(std::forward<Tuples>(tuples))...);
}

// Generates tuple<group<...>, group<...>, ...> by zipping across index sequence
template <typename... Tuples, std::size_t... Is>
auto zip_tuples(std::index_sequence<Is...>, Tuples &&...tuples) {
  return std::make_tuple(make_group_at<Is>(std::forward<Tuples>(tuples)...)...);
}

} // namespace detail

template <typename F, typename... Tuples>
auto zip_apply(F &&f, Tuples &&...tuples) -> decltype(auto) {
  static_assert(sizeof...(Tuples) >= 1, "Need at least one tuple");
  constexpr std::size_t N = std::tuple_size_v<
      std::remove_reference_t<std::tuple_element_t<0, std::tuple<Tuples...>>>>;
  static_assert(
      (... && (std::tuple_size_v<std::remove_reference_t<Tuples>> == N)),
      "All tuples must have the same size");

  auto zipped = detail::zip_tuples(std::make_index_sequence<N>{},
                                   std::forward<Tuples>(tuples)...);
  return tf::apply(std::forward<F>(f), std::move(zipped));
}
} // namespace tf
