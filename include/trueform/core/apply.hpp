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
#include <tuple>
#include <utility>

namespace tf {

/// @cond INTERNAL
namespace detail {
template <typename F, typename Tuple, std::size_t... Is>
auto apply(F &&f, std::index_sequence<Is...>, Tuple &&tuple) -> decltype(auto) {
  using std::get;
  return f(get<Is>(tuple)...);
}

} // namespace detail
/// @endcond

/// @ingroup core_algorithms
/// @brief Apply a function to tuple elements as arguments.
///
/// Unpacks a tuple and calls the function with its elements as arguments.
///
/// @tparam F The callable type.
/// @tparam Tuple The tuple type.
/// @param f The function to invoke.
/// @param tuple The tuple to unpack.
/// @return The result of calling f with tuple elements.
template <typename F, typename Tuple>
auto apply(F &&f, Tuple &&tuple) -> decltype(auto) {
  constexpr std::size_t N = std::tuple_size_v<std::decay_t<Tuple>>;
  return detail::apply(static_cast<F &&>(f), std::make_index_sequence<N>{},
                       static_cast<Tuple &&>(tuple));
}
} // namespace tf
