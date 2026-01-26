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
#include "../manifold_edge_link.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Range, typename Base> struct tag_manifold_edge_link;
template <typename Range, typename Base>
auto has_manifold_edge_link(const tag_manifold_edge_link<Range, Base> *)
    -> std::true_type;

auto has_manifold_edge_link(const void *) -> std::false_type;
} // namespace policy

/// @ingroup topology_policies
/// @brief Checks if a type has manifold edge link policy attached.
///
/// True if the type was wrapped with @ref tf::tag_manifold_edge_link().
///
/// @tparam T The type to check.
template <typename T>
inline constexpr bool has_manifold_edge_link_policy =
    decltype(policy::has_manifold_edge_link(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Range, typename Base> struct tag_manifold_edge_link : Base {
  using Base::operator=;

  tag_manifold_edge_link(
      tf::manifold_edge_link_like<Range> _manifold_edge_link_range,
      const Base &base)
      : Base{base},
        _manifold_edge_link_range{std::move(_manifold_edge_link_range)} {}

  tag_manifold_edge_link(
      tf::manifold_edge_link_like<Range> _manifold_edge_link_range, Base &&base)
      : Base{std::move(base)},
        _manifold_edge_link_range{std::move(_manifold_edge_link_range)} {}

  auto manifold_edge_link() const
      -> const tf::manifold_edge_link_like<Range> & {
    return _manifold_edge_link_range;
  }

  auto manifold_edge_link() -> tf::manifold_edge_link_like<Range> & {
    return _manifold_edge_link_range;
  }

private:
  tf::manifold_edge_link_like<Range> _manifold_edge_link_range;

  friend auto unwrap(const tag_manifold_edge_link &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_manifold_edge_link &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_manifold_edge_link &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_manifold_edge_link &val, T &&t) {
    return tag_manifold_edge_link<Range, std::decay_t<T>>{
        val._manifold_edge_link_range, static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Range, typename Base>
struct static_size<policy::tag_manifold_edge_link<Range, Base>>
    : static_size<Base> {};

/// @ingroup topology_policies
/// @brief Attaches manifold edge link data to a base type.
///
/// Creates a wrapper that carries manifold edge link information alongside
/// the original data. The result provides a `.manifold_edge_link()` accessor.
/// Use with pipe syntax: `data | tf::tag_manifold_edge_link(mel)`.
///
/// @tparam Range The manifold edge link range type.
/// @tparam Base The base type to wrap.
/// @param _manifold_edge_link_range The manifold edge link data.
/// @param base The base value to wrap.
/// @return A wrapped type with manifold edge link accessible via `.manifold_edge_link()`.
template <typename Range, typename Base>
auto tag_manifold_edge_link(
    tf::manifold_edge_link_like<Range> &&_manifold_edge_link_range,
    Base &&base) {
  if constexpr (has_manifold_edge_link_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_manifold_edge_link<Range, std::decay_t<decltype(b_base)>>{
            std::move(_manifold_edge_link_range), b_base});
  }
}

/// @overload
template <typename Index, std::size_t N, typename Base>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &_manifold_edge_link, Base &&base) {
  return tag_manifold_edge_link(
      tf::make_manifold_edge_link_like(tf::make_range(_manifold_edge_link)),
      static_cast<Base &&>(base));
}

/// @overload
template <typename Index, std::size_t N, typename Base>
auto tag_manifold_edge_link(
    const tf::manifold_edge_link<Index, N> &_manifold_edge_link, Base &&base) {
  return tag_manifold_edge_link(
      tf::make_manifold_edge_link_like(tf::make_range(_manifold_edge_link)),
      static_cast<Base &&>(base));
}

template <typename Index, std::size_t N, typename Base>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &&_manifold_edge_link,
    Base &&base) = delete;

namespace policy {
template <typename Range> struct tag_manifold_edge_link_op {
  Range manifold_edge_link_range;
};

template <typename U, typename Range>
auto operator|(U &&u, tag_manifold_edge_link_op<Range> t) {
  return tf::tag_manifold_edge_link(
      tf::make_manifold_edge_link_like(t.manifold_edge_link_range),
      static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup topology_policies
/// @brief Creates a pipe-able manifold edge link tag operator.
///
/// Returns an object that can be used with pipe syntax to attach
/// manifold edge link to a range: `data | tf::tag_manifold_edge_link(mel)`.
///
/// @tparam Range The manifold edge link range type.
/// @param _manifold_edge_link_range The manifold edge link data.
/// @return A tag operator for use with pipe syntax.
template <typename Range>
auto tag_manifold_edge_link(Range &&_manifold_edge_link_range) {
  return policy::tag_manifold_edge_link_op<Range>{
      static_cast<Range &&>(_manifold_edge_link_range)};
}

/// @overload
template <typename Index, std::size_t N>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

/// @overload
template <typename Index, std::size_t N>
auto tag_manifold_edge_link(
    const tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

template <typename Index, std::size_t N>
auto tag_manifold_edge_link(
    tf::manifold_edge_link<Index, N> &&_manifold_edge_link) = delete;

/// @ingroup topology_policies
/// @brief Creates a pipe-able tag operator for manifold edge link.
///
/// Generic overload of @ref tf::tag() that auto-detects the topology type.
/// Equivalent to `tf::tag_manifold_edge_link(_manifold_edge_link)`.
///
/// @tparam Index The index type.
/// @tparam N The static face size.
/// @param _manifold_edge_link The manifold edge link structure.
/// @return A tag operator for use with pipe syntax.
template <typename Index, std::size_t N>
auto tag(tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

/// @overload
template <typename Index, std::size_t N>
auto tag(const tf::manifold_edge_link<Index, N> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

template <typename Index, std::size_t N>
auto tag(tf::manifold_edge_link<Index, N> &&_manifold_edge_link) = delete;

/// @overload
template <typename Policy>
auto tag(tf::manifold_edge_link_like<Policy> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

/// @overload
template <typename Policy>
auto tag(const tf::manifold_edge_link_like<Policy> &_manifold_edge_link) {
  return policy::tag_manifold_edge_link_op<decltype(tf::make_range(
      _manifold_edge_link))>{tf::make_range(_manifold_edge_link)};
}

/// @overload
template <typename Policy>
auto tag(tf::manifold_edge_link_like<Policy> &&_manifold_edge_link) {
  return tag(_manifold_edge_link);
}

} // namespace tf
namespace std {
template <typename Range, typename Base>
struct tuple_size<tf::policy::tag_manifold_edge_link<Range, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Range, typename Base>
struct tuple_element<I, tf::policy::tag_manifold_edge_link<Range, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
