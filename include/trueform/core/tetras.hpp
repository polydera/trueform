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
#include "./range.hpp"
#include "./static_size.hpp"
#include "./views/blocked_range.hpp"
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup core_ranges
/// @brief Four vertex indices per tetrahedron, without adjacency.
template <typename Policy> struct tetras : Policy {
  tetras(const Policy &r) : Policy{r} {}
  tetras(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const tetras<Policy> &tet) -> decltype(auto) {
  return static_cast<const Policy &>(tet);
}

template <typename Policy> auto unwrap(tetras<Policy> &tet) -> decltype(auto) {
  return static_cast<Policy &>(tet);
}

template <typename Policy> auto unwrap(tetras<Policy> &&tet) -> decltype(auto) {
  return static_cast<Policy &&>(tet);
}

template <typename Policy, typename T>
auto wrap_like(const tetras<Policy> &, T &&t) {
  return tetras<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(tetras<Policy> &, T &&t) {
  return tetras<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tetras<Policy> &&, T &&t) {
  return tetras<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_ranges
/// @brief Borrow blocks of four indices, or a flat range whose size is a
/// multiple of four.
template <typename Range> auto make_tetras(Range &&r) {
  auto r0 = tf::make_range(r);
  if constexpr (tf::static_size_v<decltype(r0[0])> == 4) {
    return tf::tetras<decltype(r0)>{r0};
  } else {
    static_assert(std::is_integral_v<std::decay_t<decltype(r0[0])>>,
                  "Tetras require flat indices or blocks of four indices");
    return make_tetras(tf::make_blocked_range<4>(r0));
  }
}

template <typename Range> auto make_tetras(tetras<Range> r) -> tetras<Range> {
  return r;
}

template <typename Policy> auto make_view(const tf::tetras<Policy> &obj) {
  return obj;
}

template <typename Policy> auto make_view(tf::tetras<Policy> &obj) {
  return obj;
}

template <typename Policy> auto make_view(tf::tetras<Policy> &&obj) {
  return std::move(obj);
}

} // namespace tf
