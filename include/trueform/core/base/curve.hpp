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
#include "../coordinate_type.hpp"
#include "../coordinate_dims.hpp"
#include "../static_size.hpp"
#include "../views/indirect_range.hpp"
namespace tf::core {
template <std::size_t V, typename Policy>
class curve : public tf::core::assignable_range<V, Policy> {
private:
  using base_t = tf::core::assignable_range<V, Policy>;

public:
  curve(const Policy &policy) : base_t{policy} {}
  curve(Policy &&policy) : base_t{std::move(policy)} {}
  curve() = default;
  curve(const curve &) = default;
  curve(curve &&) = default;
  auto operator=(const curve &) -> curve & = default;
  auto operator=(curve &&) -> curve & = default;
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
class curve<tf::dynamic_size, Policy>
    : public tf::core::assignable_range<tf::dynamic_size, Policy> {
private:
  using base_t = tf::core::assignable_range<tf::dynamic_size, Policy>;

public:
  curve(const Policy &policy) : base_t{policy} {}
  curve(Policy &&policy) : base_t{std::move(policy)} {}
  curve() = default;
  curve(const curve &) = default;
  curve(curve &&) = default;
  auto operator=(const curve &) -> curve & = default;
  auto operator=(curve &&) -> curve & = default;
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
auto is_curve_impl(const curve<V, Base> *) -> std::true_type;
auto is_curve_impl(const void *) -> std::false_type;

template <typename T>
inline constexpr bool is_curve = decltype(is_curve_impl(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

template <typename Range> auto make_curve(Range &&r) {
  if constexpr (is_curve<Range>)
    return r;
  else
    return curve<tf::static_size_v<Range>, std::decay_t<Range>>{
        static_cast<Range &&>(r)};
}
template <std::size_t V, typename Range> auto make_curve(Range &&r) {
  if constexpr (is_curve<Range>)
    return r;
  else
    return curve<V, std::decay_t<Range>>{static_cast<Range &&>(r)};
}

template <std::size_t V, typename Range0, typename Range1>
auto make_curve(Range0 &&ids, Range1 &&data) {
  return views::make_indirect_range(
      core::make_curve<V>(views::make_indirect_range_base(
          static_cast<Range0 &&>(ids), static_cast<Range1 &&>(data))));
}

template <typename Range0, typename Range1>
auto make_curve(Range0 &&ids, Range1 &&data) {
  return core::make_curve<tf::static_size_v<Range0>>(
      static_cast<Range0 &&>(ids), static_cast<Range1 &&>(data));
}
} // namespace tf::core

namespace tf {
template <std::size_t V, typename Policy>
struct static_size<tf::core::curve<V, Policy>>
    : std::integral_constant<std::size_t, V> {};
} // namespace tf
namespace std {

template <std::size_t I, std::size_t V, typename Policy>
struct tuple_element<I, tf::core::curve<V, Policy>> : tuple_element<I, Policy> {
};

template <std::size_t V, typename Policy>
struct tuple_size<tf::core::curve<V, Policy>>
    : std::integral_constant<std::size_t, V> {};

} // namespace std
