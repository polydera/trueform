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
namespace tf::core {
template <std::size_t V, typename Policy>
class poly : public tf::core::assignable_range<V, Policy> {
private:
  using base_t = tf::core::assignable_range<V, Policy>;

public:
  poly(const Policy &policy) : base_t{policy} {}
  poly(Policy &&policy) : base_t{std::move(policy)} {}
  poly() = default;
  poly(const poly &) = default;
  poly(poly &&) = default;
  auto operator=(const poly &) -> poly & = default;
  auto operator=(poly &&) -> poly & = default;
  using coordinate_type = tf::coordinate_type<typename Policy::value_type>;
  using coordinate_dims = tf::coordinate_dims<typename Policy::value_type>;
  using base_t::base_t;
  using base_t::operator=;
  using base_t::operator[];
  using base_t::begin;
  using base_t::end;

  constexpr auto size() const { return V; }
};

template <typename Policy>
class poly<tf::dynamic_size, Policy>
    : public tf::core::assignable_range<tf::dynamic_size, Policy> {
private:
  using base_t = tf::core::assignable_range<tf::dynamic_size, Policy>;

public:
  poly(const Policy &policy) : base_t{policy} {}
  poly(Policy &&policy) : base_t{std::move(policy)} {}
  poly() = default;
  poly(const poly &) = default;
  poly(poly &&) = default;
  auto operator=(const poly &) -> poly & = default;
  auto operator=(poly &&) -> poly & = default;
  using coordinate_type = tf::coordinate_type<typename Policy::value_type>;
  using coordinate_dims = tf::coordinate_dims<typename Policy::value_type>;
  using base_t::base_t;
  using base_t::operator=;
  using base_t::operator[];
  using base_t::begin;
  using base_t::end;
  using base_t::size;
};

template <std::size_t V, typename Base>
auto is_poly_impl(const poly<V, Base> *) -> std::true_type;
auto is_poly_impl(const void *) -> std::false_type;

template <typename T>
inline constexpr bool is_poly = decltype(is_poly_impl(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

template <typename Range> auto make_poly(Range &&r) {
  if constexpr (is_poly<Range>)
    return r;
  else
    return poly<tf::static_size_v<Range>, std::decay_t<Range>>{
        static_cast<Range &&>(r)};
}
template <std::size_t V, typename Range> auto make_poly(Range &&r) {
  if constexpr (is_poly<Range>)
    return r;
  else
    return poly<V, std::decay_t<Range>>{static_cast<Range &&>(r)};
}

template <std::size_t V, typename Range0, typename Range1>
auto make_poly(Range0 &&ids, Range1 &&data) {
  return views::make_indirect_range(
      make_poly<V>(views::make_indirect_range_base(
          static_cast<Range0 &&>(ids), static_cast<Range1 &&>(data))));
}

template <typename Range0, typename Range1>
auto make_poly(Range0 &&ids, Range1 &&data) {
  return make_poly<tf::static_size_v<Range0>>(static_cast<Range0 &&>(ids),
                                              static_cast<Range1 &&>(data));
}
} // namespace tf::core

namespace tf {
template <std::size_t V, typename Policy>
struct static_size<tf::core::poly<V, Policy>>
    : std::integral_constant<std::size_t, V> {};
} // namespace tf
namespace std {

template <std::size_t I, std::size_t V, typename Policy>
struct tuple_element<I, tf::core::poly<V, Policy>> : tuple_element<I, Policy> {
};

template <std::size_t V, typename Policy>
struct tuple_size<tf::core::poly<V, Policy>>
    : std::integral_constant<std::size_t, V> {};

} // namespace std
