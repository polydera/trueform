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
#include "../edge_membership.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Range, typename Base> struct tag_edge_membership;
template <typename Range, typename Base>
auto has_edge_membership(const tag_edge_membership<Range, Base> *)
    -> std::true_type;

auto has_edge_membership(const void *) -> std::false_type;
} // namespace policy

/// @ingroup topology_policies
/// @brief Checks if a type has edge membership policy attached.
///
/// True if the type was wrapped with @ref tf::tag_edge_membership().
///
/// @tparam T The type to check.
template <typename T>
inline constexpr bool has_edge_membership_policy =
    decltype(policy::has_edge_membership(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Range, typename Base> struct tag_edge_membership : Base {
  using Base::operator=;

  tag_edge_membership(tf::edge_membership_like<Range> _edge_membership_range,
                      const Base &base)
      : Base{base}, _edge_membership_range{std::move(_edge_membership_range)} {}

  tag_edge_membership(tf::edge_membership_like<Range> _edge_membership_range,
                      Base &&base)
      : Base{std::move(base)},
        _edge_membership_range{std::move(_edge_membership_range)} {}

  auto edge_membership() const -> const tf::edge_membership_like<Range> & {
    return _edge_membership_range;
  }

  auto edge_membership() -> tf::edge_membership_like<Range> & {
    return _edge_membership_range;
  }

private:
  tf::edge_membership_like<Range> _edge_membership_range;

  friend auto unwrap(const tag_edge_membership &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_edge_membership &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_edge_membership &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_edge_membership &val, T &&t) {
    return tag_edge_membership<Range, std::decay_t<T>>{
        val._edge_membership_range, static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_edge_membership<Range, Base>>
    : static_size<Base> {};

/// @ingroup topology_policies
/// @brief Attaches edge membership data to a base type.
///
/// Creates a wrapper that carries edge membership information alongside
/// the original data. The result provides a `.edge_membership()` accessor.
/// Use with pipe syntax: `data | tf::tag_edge_membership(em)`.
///
/// @tparam Range The edge membership range type.
/// @tparam Base The base type to wrap.
/// @param _edge_membership_range The edge membership data.
/// @param base The base value to wrap.
/// @return A wrapped type with edge membership accessible via `.edge_membership()`.
template <typename Range, typename Base>
auto tag_edge_membership(
    tf::edge_membership_like<Range> &&_edge_membership_range, Base &&base) {
  if constexpr (has_edge_membership_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_edge_membership<Range, std::decay_t<decltype(b_base)>>{
            std::move(_edge_membership_range), b_base});
  }
}

/// @overload
template <typename Index, typename Base>
auto tag_edge_membership(tf::edge_membership<Index> &_edge_membership,
                         Base &&base) {
  return tag_edge_membership(
      tf::make_edge_membership_like(tf::make_range(_edge_membership)),
      static_cast<Base &&>(base));
}

/// @overload
template <typename Index, typename Base>
auto tag_edge_membership(const tf::edge_membership<Index> &_edge_membership,
                         Base &&base) {
  return tag_edge_membership(
      tf::make_edge_membership_like(tf::make_range(_edge_membership)),
      static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_edge_membership(tf::edge_membership<Index> &&_edge_membership,
                         Base &&base) = delete;

namespace policy {
template <typename Range> struct tag_edge_membership_op {
  Range edge_membership_range;
};

template <typename U, typename Range>
auto operator|(U &&u, tag_edge_membership_op<Range> t) {
  return tf::tag_edge_membership(
      tf::make_edge_membership_like(t.edge_membership_range),
      static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup topology_policies
/// @brief Creates a pipe-able edge membership tag operator.
///
/// Returns an object that can be used with pipe syntax to attach
/// edge membership to a range: `data | tf::tag_edge_membership(em)`.
///
/// @tparam Range The edge membership range type.
/// @param _edge_membership_range The edge membership data.
/// @return A tag operator for use with pipe syntax.
template <typename Range>
auto tag_edge_membership(Range &&_edge_membership_range) {
  return policy::tag_edge_membership_op<Range>{
      static_cast<Range &&>(_edge_membership_range)};
}

/// @overload
template <typename Index>
auto tag_edge_membership(tf::edge_membership<Index> &_edge_membership) {
  return policy::tag_edge_membership_op<decltype(tf::make_range(
      _edge_membership))>{tf::make_range(_edge_membership)};
}

/// @overload
template <typename Index>
auto tag_edge_membership(const tf::edge_membership<Index> &_edge_membership) {
  return policy::tag_edge_membership_op<decltype(tf::make_range(
      _edge_membership))>{tf::make_range(_edge_membership)};
}

template <typename Index>
auto tag_edge_membership(tf::edge_membership<Index> &&_edge_membership) =
    delete;

/// @ingroup topology_policies
/// @brief Creates a pipe-able tag operator for edge membership.
///
/// Generic overload of @ref tf::tag() that auto-detects the topology type.
/// Equivalent to `tf::tag_edge_membership(_edge_membership)`.
///
/// @tparam Index The index type.
/// @param _edge_membership The edge membership structure.
/// @return A tag operator for use with pipe syntax.
template <typename Index>
auto tag(tf::edge_membership<Index> &_edge_membership) {
  return policy::tag_edge_membership_op<decltype(tf::make_range(
      _edge_membership))>{tf::make_range(_edge_membership)};
}

/// @overload
template <typename Index>
auto tag(const tf::edge_membership<Index> &_edge_membership) {
  return policy::tag_edge_membership_op<decltype(tf::make_range(
      _edge_membership))>{tf::make_range(_edge_membership)};
}

template <typename Index>
auto tag(tf::edge_membership<Index> &&_edge_membership) = delete;

/// @overload
template <typename Policy>
auto tag(tf::edge_membership_like<Policy> &_edge_membership) {
  return policy::tag_edge_membership_op<decltype(tf::make_range(
      _edge_membership))>{tf::make_range(_edge_membership)};
}

/// @overload
template <typename Policy>
auto tag(const tf::edge_membership_like<Policy> &_edge_membership) {
  return policy::tag_edge_membership_op<decltype(tf::make_range(
      _edge_membership))>{tf::make_range(_edge_membership)};
}

/// @overload
template <typename Policy>
auto tag(tf::edge_membership_like<Policy> &&_edge_membership) {
  return tag(_edge_membership);
}

} // namespace tf
namespace std {
template <typename Range, typename Base>
struct tuple_size<tf::policy::tag_edge_membership<Range, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Range, typename Base>
struct tuple_element<I, tf::policy::tag_edge_membership<Range, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
