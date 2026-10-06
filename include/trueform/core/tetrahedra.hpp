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
#include "./base/soup.hpp"
#include "./base/tetrahedra.hpp"
#include "./coordinate_dims.hpp"
#include "./form.hpp"
#include "./points.hpp"
#include "./range.hpp"
#include "./tetras.hpp"
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup core_ranges
/// @brief A policy-composable view of tetrahedra, from tetras and points or a
/// soup.
template <typename Policy> struct tetrahedra : form<3, Policy> {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "Tetrahedra require three-dimensional points");
  using base = form<3, Policy>;
  tetrahedra(const Policy &r) : base{r} {}
  tetrahedra(Policy &&r) : base{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const tetrahedra<Policy> &tet) -> decltype(auto) {
  return static_cast<const Policy &>(tet);
}

template <typename Policy>
auto unwrap(tetrahedra<Policy> &tet) -> decltype(auto) {
  return static_cast<Policy &>(tet);
}

template <typename Policy>
auto unwrap(tetrahedra<Policy> &&tet) -> decltype(auto) {
  return static_cast<Policy &&>(tet);
}

template <typename Policy, typename T>
auto wrap_like(const tetrahedra<Policy> &, T &&t) {
  return tetrahedra<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tetrahedra<Policy> &, T &&t) {
  return tetrahedra<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tetrahedra<Policy> &&, T &&t) {
  return tetrahedra<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup core_ranges
/// @brief Borrow four-index connectivity and point coordinates.
template <typename Range0, typename Range1>
auto make_tetrahedra(Range0 &&tetras, Range1 &&points) {
  auto r0 = tf::make_tetras(tetras);
  auto r1 = tf::make_points(points);
  return tetrahedra<core::tetrahedra<decltype(r0), decltype(r1)>>{
      core::tetrahedra<decltype(r0), decltype(r1)>{r0, r1}};
}

template <typename Range>
auto make_tetrahedra(tetrahedra<Range> p) -> tetrahedra<Range> {
  return p;
}

/// @ingroup core_ranges
/// @brief Borrow a range of tetrahedron primitives without shared
/// connectivity.
template <typename Range> auto make_tetrahedra(Range &&r) {
  auto tets = tf::make_range(r);
  return tetrahedra<core::soup<decltype(tets)>>{
      core::soup<decltype(tets)>{tets}};
}

template <typename Policy> auto make_view(const tf::tetrahedra<Policy> &obj) {
  return obj;
}

template <typename Policy> auto make_view(tf::tetrahedra<Policy> &obj) {
  return obj;
}

template <typename Policy> auto make_view(tf::tetrahedra<Policy> &&obj) {
  return std::move(obj);
}

template <typename T>
inline constexpr bool has_tetras = decltype(core::has_tetras_policy(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

} // namespace tf
