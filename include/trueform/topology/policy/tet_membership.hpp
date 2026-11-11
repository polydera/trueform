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
#include "../../core/policy/unwrap.hpp"
#include "../tet_membership.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Range, typename Base> struct tag_tet_membership;
template <typename Range, typename Base>
auto has_tet_membership(const tag_tet_membership<Range, Base> *)
    -> std::true_type;

auto has_tet_membership(const void *) -> std::false_type;
} // namespace policy

/// @ingroup topology_policies
/// @brief Checks if a type has tet membership policy attached.
///
/// True if the type was wrapped with @ref tf::tag_tet_membership().
///
/// @tparam T The type to check.
template <typename T>
inline constexpr bool has_tet_membership_policy =
    decltype(policy::has_tet_membership(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Range, typename Base> struct tag_tet_membership : Base {
  using Base::operator=;

  tag_tet_membership(tf::tet_membership_like<Range> _tet_membership_range,
                     const Base &base)
      : Base{base}, _tet_membership_range{std::move(_tet_membership_range)} {}

  tag_tet_membership(tf::tet_membership_like<Range> _tet_membership_range,
                     Base &&base)
      : Base{std::move(base)},
        _tet_membership_range{std::move(_tet_membership_range)} {}

  auto tet_membership() const -> const tf::tet_membership_like<Range> & {
    return _tet_membership_range;
  }

  auto tet_membership() -> tf::tet_membership_like<Range> & {
    return _tet_membership_range;
  }

private:
  tf::tet_membership_like<Range> _tet_membership_range;

  friend auto unwrap(const tag_tet_membership &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_tet_membership &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_tet_membership &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_tet_membership &val, T &&t) {
    return tag_tet_membership<Range, std::decay_t<T>>{val._tet_membership_range,
                                                      static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_tet_membership<Range, Base>>
    : static_size<Base> {};

/// @ingroup topology_policies
/// @brief Attaches tet membership data to a base type.
///
/// Creates a wrapper that carries tet membership information alongside
/// the original data. The result provides a `.tet_membership()` accessor.
/// Use with pipe syntax: `data | tf::tag_tet_membership(membership)`.
///
/// @tparam Range The tet membership range type.
/// @tparam Base The base type to wrap.
/// @param _tet_membership_range The tet membership data.
/// @param base The base value to wrap.
/// @return A wrapped type with tet membership accessible via
/// `.tet_membership()`.
template <typename Range, typename Base>
auto tag_tet_membership(tf::tet_membership_like<Range> &&_tet_membership_range,
                        Base &&base) {
  if constexpr (has_tet_membership_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_tet_membership<Range, std::decay_t<decltype(b_base)>>{
                  std::move(_tet_membership_range), b_base});
  }
}

/// @overload
template <typename Index, typename Base>
auto tag_tet_membership(tf::tet_membership<Index> &_tet_membership,
                        Base &&base) {
  return tag_tet_membership(
      tf::make_tet_membership_like(tf::make_range(_tet_membership)),
      static_cast<Base &&>(base));
}

/// @overload
template <typename Index, typename Base>
auto tag_tet_membership(const tf::tet_membership<Index> &_tet_membership,
                        Base &&base) {
  return tag_tet_membership(
      tf::make_tet_membership_like(tf::make_range(_tet_membership)),
      static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_tet_membership(tf::tet_membership<Index> &&_tet_membership,
                        Base &&base) = delete;

namespace policy {
template <typename Range> struct tag_tet_membership_op {
  Range tet_membership_range;
};

template <typename U, typename Range>
auto operator|(U &&u, tag_tet_membership_op<Range> t) {
  return tf::tag_tet_membership(
      tf::make_tet_membership_like(t.tet_membership_range),
      static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup topology_policies
/// @brief Creates a pipe-able tet membership tag operator.
///
/// Returns an object that can be used with pipe syntax to attach
/// tet membership to a range: `data | tf::tag_tet_membership(membership)`.
///
/// @tparam Range The tet membership range type.
/// @param _tet_membership_range The tet membership data.
/// @return A tag operator for use with pipe syntax.
template <typename Range>
auto tag_tet_membership(Range &&_tet_membership_range) {
  return policy::tag_tet_membership_op<Range>{
      static_cast<Range &&>(_tet_membership_range)};
}

/// @overload
template <typename Index>
auto tag_tet_membership(tf::tet_membership<Index> &_tet_membership) {
  return policy::tag_tet_membership_op<decltype(tf::make_range(
      _tet_membership))>{tf::make_range(_tet_membership)};
}

/// @overload
template <typename Index>
auto tag_tet_membership(const tf::tet_membership<Index> &_tet_membership) {
  return policy::tag_tet_membership_op<decltype(tf::make_range(
      _tet_membership))>{tf::make_range(_tet_membership)};
}

template <typename Index>
auto tag_tet_membership(tf::tet_membership<Index> &&_tet_membership) = delete;

/// @ingroup topology_policies
/// @brief Creates a pipe-able tag operator for tet membership.
///
/// Generic overload of @ref tf::tag() that auto-detects the topology type.
/// Equivalent to `tf::tag_tet_membership(_tet_membership)`.
///
/// @tparam Index The index type.
/// @param _tet_membership The tet membership structure.
/// @return A tag operator for use with pipe syntax.
template <typename Index> auto tag(tf::tet_membership<Index> &_tet_membership) {
  return policy::tag_tet_membership_op<decltype(tf::make_range(
      _tet_membership))>{tf::make_range(_tet_membership)};
}

/// @overload
template <typename Index>
auto tag(const tf::tet_membership<Index> &_tet_membership) {
  return policy::tag_tet_membership_op<decltype(tf::make_range(
      _tet_membership))>{tf::make_range(_tet_membership)};
}

template <typename Index>
auto tag(tf::tet_membership<Index> &&_tet_membership) = delete;

/// @overload
template <typename Policy>
auto tag(tf::tet_membership_like<Policy> &_tet_membership) {
  return policy::tag_tet_membership_op<decltype(tf::make_range(
      _tet_membership))>{tf::make_range(_tet_membership)};
}

/// @overload
template <typename Policy>
auto tag(const tf::tet_membership_like<Policy> &_tet_membership) {
  return policy::tag_tet_membership_op<decltype(tf::make_range(
      _tet_membership))>{tf::make_range(_tet_membership)};
}

/// @overload
template <typename Policy>
auto tag(tf::tet_membership_like<Policy> &&_tet_membership) {
  return tag(_tet_membership);
}

} // namespace tf
namespace std {
template <typename Range, typename Base>
struct tuple_size<tf::policy::tag_tet_membership<Range, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Range, typename Base>
struct tuple_element<I, tf::policy::tag_tet_membership<Range, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
