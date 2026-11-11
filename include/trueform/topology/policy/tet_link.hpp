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
#include "../tet_link.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Range, typename Base> struct tag_tet_link;
template <typename Range, typename Base>
auto has_tet_link(const tag_tet_link<Range, Base> *) -> std::true_type;

auto has_tet_link(const void *) -> std::false_type;
} // namespace policy

/// @ingroup topology_policies
/// @brief Checks if a type has tet link policy attached.
///
/// True if the type was wrapped with @ref tf::tag_tet_link().
///
/// @tparam T The type to check.
template <typename T>
inline constexpr bool has_tet_link_policy = decltype(policy::has_tet_link(
    static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Range, typename Base> struct tag_tet_link : Base {
  using Base::operator=;

  tag_tet_link(tf::tet_link_like<Range> _tet_link_range, const Base &base)
      : Base{base}, _tet_link_range{std::move(_tet_link_range)} {}

  tag_tet_link(tf::tet_link_like<Range> _tet_link_range, Base &&base)
      : Base{std::move(base)}, _tet_link_range{std::move(_tet_link_range)} {}

  auto tet_link() const -> const tf::tet_link_like<Range> & {
    return _tet_link_range;
  }

  auto tet_link() -> tf::tet_link_like<Range> & { return _tet_link_range; }

private:
  tf::tet_link_like<Range> _tet_link_range;

  friend auto unwrap(const tag_tet_link &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_tet_link &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_tet_link &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T> friend auto wrap_like(const tag_tet_link &val, T &&t) {
    return tag_tet_link<Range, std::decay_t<T>>{val._tet_link_range,
                                                static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_tet_link<Range, Base>> : static_size<Base> {};

/// @ingroup topology_policies
/// @brief Attaches tet link data to a base type.
///
/// Creates a wrapper that carries tet link information alongside
/// the original data. The result provides a `.tet_link()` accessor.
/// Use with pipe syntax: `data | tf::tag_tet_link(link)`.
///
/// @tparam Range The tet link range type.
/// @tparam Base The base type to wrap.
/// @param _tet_link_range The tet link data.
/// @param base The base value to wrap.
/// @return A wrapped type with tet link accessible via `.tet_link()`.
template <typename Range, typename Base>
auto tag_tet_link(tf::tet_link_like<Range> &&_tet_link_range, Base &&base) {
  if constexpr (has_tet_link_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base, policy::tag_tet_link<Range, std::decay_t<decltype(b_base)>>{
                  std::move(_tet_link_range), b_base});
  }
}

/// @overload
template <typename Index, typename Base>
auto tag_tet_link(tf::tet_link<Index> &_tet_link, Base &&base) {
  return tag_tet_link(tf::make_tet_link_like(tf::make_range(_tet_link)),
                      static_cast<Base &&>(base));
}

/// @overload
template <typename Index, typename Base>
auto tag_tet_link(const tf::tet_link<Index> &_tet_link, Base &&base) {
  return tag_tet_link(tf::make_tet_link_like(tf::make_range(_tet_link)),
                      static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_tet_link(tf::tet_link<Index> &&_tet_link, Base &&base) = delete;

namespace policy {
template <typename Range> struct tag_tet_link_op {
  Range tet_link_range;
};

template <typename U, typename Range>
auto operator|(U &&u, tag_tet_link_op<Range> t) {
  return tf::tag_tet_link(tf::make_tet_link_like(t.tet_link_range),
                          static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup topology_policies
/// @brief Creates a pipe-able tet link tag operator.
///
/// Returns an object that can be used with pipe syntax to attach
/// tet link to a range: `data | tf::tag_tet_link(link)`.
///
/// @tparam Range The tet link range type.
/// @param _tet_link_range The tet link data.
/// @return A tag operator for use with pipe syntax.
template <typename Range> auto tag_tet_link(Range &&_tet_link_range) {
  return policy::tag_tet_link_op<Range>{static_cast<Range &&>(_tet_link_range)};
}

/// @overload
template <typename Index> auto tag_tet_link(tf::tet_link<Index> &_tet_link) {
  return policy::tag_tet_link_op<decltype(tf::make_range(_tet_link))>{
      tf::make_range(_tet_link)};
}

/// @overload
template <typename Index>
auto tag_tet_link(const tf::tet_link<Index> &_tet_link) {
  return policy::tag_tet_link_op<decltype(tf::make_range(_tet_link))>{
      tf::make_range(_tet_link)};
}

template <typename Index>
auto tag_tet_link(tf::tet_link<Index> &&_tet_link) = delete;

/// @ingroup topology_policies
/// @brief Creates a pipe-able tag operator for tet link.
///
/// Generic overload of @ref tf::tag() that auto-detects the topology type.
/// Equivalent to `tf::tag_tet_link(_tet_link)`.
///
/// @tparam Index The index type.
/// @param _tet_link The tet link structure.
/// @return A tag operator for use with pipe syntax.
template <typename Index> auto tag(tf::tet_link<Index> &_tet_link) {
  return policy::tag_tet_link_op<decltype(tf::make_range(_tet_link))>{
      tf::make_range(_tet_link)};
}

/// @overload
template <typename Index> auto tag(const tf::tet_link<Index> &_tet_link) {
  return policy::tag_tet_link_op<decltype(tf::make_range(_tet_link))>{
      tf::make_range(_tet_link)};
}

template <typename Index> auto tag(tf::tet_link<Index> &&_tet_link) = delete;

/// @overload
template <typename Policy> auto tag(tf::tet_link_like<Policy> &_tet_link) {
  return policy::tag_tet_link_op<decltype(tf::make_range(_tet_link))>{
      tf::make_range(_tet_link)};
}

/// @overload
template <typename Policy>
auto tag(const tf::tet_link_like<Policy> &_tet_link) {
  return policy::tag_tet_link_op<decltype(tf::make_range(_tet_link))>{
      tf::make_range(_tet_link)};
}

/// @overload
template <typename Policy> auto tag(tf::tet_link_like<Policy> &&_tet_link) {
  return tag(_tet_link);
}

} // namespace tf
namespace std {
template <typename Range, typename Base>
struct tuple_size<tf::policy::tag_tet_link<Range, Base>> : tuple_size<Base> {};

template <std::size_t I, typename Range, typename Base>
struct tuple_element<I, tf::policy::tag_tet_link<Range, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
