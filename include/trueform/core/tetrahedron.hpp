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
#include "./base/tetra.hpp"
#include "./coordinate_dims.hpp"
#include "./static_size.hpp"
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup core_primitives
/// @brief Four 3D points with tetrahedral semantics and composable storage
/// policies.
template <typename Policy> class tetrahedron : public Policy {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "Tetrahedra require three-dimensional points");

private:
  using base_t = Policy;

public:
  tetrahedron(const Policy &policy) : base_t{policy} {}
  tetrahedron(Policy &&policy) : base_t{std::move(policy)} {}
  tetrahedron() = default;
  using base_t::base_t;
  using base_t::operator=;
  using base_t::operator[];
  using base_t::begin;
  using base_t::end;
  using base_t::size;

  friend auto unwrap(const tetrahedron &tetra) -> decltype(auto) {
    return static_cast<const Policy &>(tetra);
  }

  friend auto unwrap(tetrahedron &tetra) -> decltype(auto) {
    return static_cast<Policy &>(tetra);
  }

  friend auto unwrap(tetrahedron &&tetra) -> decltype(auto) {
    return static_cast<Policy &&>(tetra);
  }

  template <typename T> friend auto wrap_like(const tetrahedron &, T &&t) {
    return tetrahedron<std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(tetrahedron &, T &&t) {
    return tetrahedron<std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T> friend auto wrap_like(tetrahedron &&, T &&t) {
    return tetrahedron<std::decay_t<T>>{static_cast<T &&>(t)};
  }
};

template <std::size_t I, typename Policy>
auto get(const tf::tetrahedron<Policy> &t) -> decltype(auto) {
  return t[I];
}

template <std::size_t I, typename Policy>
auto get(tf::tetrahedron<Policy> &t) -> decltype(auto) {
  return t[I];
}

template <std::size_t I, typename Policy>
auto get(tf::tetrahedron<Policy> &&t) -> decltype(auto) {
  return t[I];
}

} // namespace tf

namespace std {

template <std::size_t I, typename Policy>
struct tuple_element<I, tf::tetrahedron<Policy>> {
  using type =
      typename iterator_traits<decltype(declval<Policy>().begin())>::value_type;
};

template <typename Policy>
struct tuple_size<tf::tetrahedron<Policy>>
    : std::integral_constant<std::size_t, 4> {};

} // namespace std

namespace tf {

template <typename Policy>
struct static_size<tf::tetrahedron<Policy>>
    : std::integral_constant<std::size_t, 4> {};

/// @ingroup core_primitives
/// @brief Borrow exactly four indices and their referenced points.
template <typename Range0, typename Range1>
auto make_tetrahedron(Range0 &&ids, Range1 &&points) {
  auto policy = tf::core::make_tetra(static_cast<Range0 &&>(ids),
                                     static_cast<Range1 &&>(points));
  return tf::tetrahedron<decltype(policy)>(std::move(policy));
}

/// @ingroup core_primitives
/// @brief Construct from exactly four points, owned or borrowed according to
/// the policy.
template <typename Range> auto make_tetrahedron(Range &&points) {
  auto policy = tf::core::make_tetra(static_cast<Range &&>(points));
  return tf::tetrahedron<decltype(policy)>(std::move(policy));
}

} // namespace tf
