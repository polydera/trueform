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
#include "../assignable_range.hpp"
#include "../coordinate_dims.hpp"
#include "../coordinate_type.hpp"
#include "../static_size.hpp"
#include "../views/indirect_range.hpp"
#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::core {
template <typename Policy>
class tetra : public tf::core::assignable_range<4, Policy> {
private:
  using base_t = tf::core::assignable_range<4, Policy>;

public:
  static_assert(tf::static_size_v<Policy> == 4 ||
                tf::static_size_v<Policy> == tf::dynamic_size);
  tetra(const Policy &policy) : base_t{policy} {}
  tetra(Policy &&policy) : base_t{std::move(policy)} {}
  tetra() = default;
  tetra(const tetra &) = default;
  tetra(tetra &&) = default;
  auto operator=(const tetra &) -> tetra & = default;
  auto operator=(tetra &&) -> tetra & = default;
  using coordinate_type = tf::coordinate_type<typename Policy::value_type>;
  using coordinate_dims = tf::coordinate_dims<typename Policy::value_type>;
  using base_t::base_t;
  using base_t::operator=;

  auto operator[](std::size_t i) -> decltype(auto) {
    return base_t::operator[](i);
  }

  auto operator[](std::size_t n) const -> decltype(auto) {
    return base_t::operator[](n);
  }

  auto begin() const { return base_t::begin(); }

  auto begin() { return base_t::begin(); }

  auto end() const { return base_t::end(); }

  auto end() { return base_t::end(); }

  constexpr auto size() const { return 4; }
};

template <typename Base>
auto is_tetra_impl(const tetra<Base> *) -> std::true_type;
auto is_tetra_impl(const void *) -> std::false_type;

template <typename T>
inline constexpr bool is_tetra = decltype(is_tetra_impl(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

template <typename Policy> auto make_tetra(Policy &&policy) {
  if constexpr (is_tetra<Policy>)
    return policy;
  else
    return core::tetra<std::decay_t<Policy>>{static_cast<Policy &&>(policy)};
}

template <typename Range0, typename Range1>
auto make_tetra(Range0 &&ids, Range1 &&data) {
  return views::make_indirect_range(make_tetra(views::make_indirect_range_base(
      static_cast<Range0 &&>(ids), static_cast<Range1 &&>(data))));
}
} // namespace tf::core

namespace tf {
template <typename Policy>
struct static_size<tf::core::tetra<Policy>>
    : std::integral_constant<std::size_t, 4> {};
} // namespace tf
namespace std {
template <typename Policy>
struct tuple_size<tf::core::tetra<Policy>> : integral_constant<size_t, 4> {};
template <std::size_t I, typename Policy>
struct tuple_element<I, tf::core::tetra<Policy>> {
  using type =
      typename iterator_traits<decltype(declval<Policy>().begin())>::value_type;
};
} // namespace std
