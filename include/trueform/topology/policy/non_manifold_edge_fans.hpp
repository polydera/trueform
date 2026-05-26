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
#include "../non_manifold_edge_fans.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Policy, typename Base> struct tag_non_manifold_edge_fans;

template <typename Policy, typename Base>
auto has_non_manifold_edge_fans(const tag_non_manifold_edge_fans<Policy, Base> *)
    -> std::true_type;

auto has_non_manifold_edge_fans(const void *) -> std::false_type;

} // namespace policy

/// @ingroup topology_policies
/// @brief Checks if a type has non-manifold edge fans policy attached.
///
/// True if the type was wrapped with @ref tf::tag_non_manifold_edge_fans().
///
/// @tparam T The type to check.
template <typename T>
inline constexpr bool has_non_manifold_edge_fans_policy =
    decltype(policy::has_non_manifold_edge_fans(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Policy, typename Base>
struct tag_non_manifold_edge_fans : Base {
  using Base::operator=;

  tag_non_manifold_edge_fans(tf::non_manifold_edge_fans_like<Policy> _fans,
                             const Base &base)
      : Base{base}, _fans{std::move(_fans)} {}

  tag_non_manifold_edge_fans(tf::non_manifold_edge_fans_like<Policy> _fans,
                             Base &&base)
      : Base{std::move(base)}, _fans{std::move(_fans)} {}

  auto non_manifold_edge_fans() const
      -> const tf::non_manifold_edge_fans_like<Policy> & {
    return _fans;
  }

  auto non_manifold_edge_fans() -> tf::non_manifold_edge_fans_like<Policy> & {
    return _fans;
  }

private:
  tf::non_manifold_edge_fans_like<Policy> _fans;

  friend auto unwrap(const tag_non_manifold_edge_fans &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_non_manifold_edge_fans &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_non_manifold_edge_fans &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_non_manifold_edge_fans &val, T &&t) {
    return tag_non_manifold_edge_fans<Policy, std::decay_t<T>>{
        val._fans, static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Policy, typename Base>
struct static_size<policy::tag_non_manifold_edge_fans<Policy, Base>>
    : static_size<Base> {};

/// @ingroup topology_policies
/// @brief Attaches non-manifold edge fans to a base type.
///
/// Creates a wrapper that carries the fans view alongside the base data.
/// The result provides a `.non_manifold_edge_fans()` accessor. Use with
/// pipe syntax: `data | tf::tag_non_manifold_edge_fans(fans)`.
///
/// @tparam Policy The non-manifold edge fans policy type.
/// @tparam Base The base type to wrap.
template <typename Policy, typename Base>
auto tag_non_manifold_edge_fans(
    tf::non_manifold_edge_fans_like<Policy> &&_fans, Base &&base) {
  if constexpr (has_non_manifold_edge_fans_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_non_manifold_edge_fans<Policy,
                                           std::decay_t<decltype(b_base)>>{
            std::move(_fans), b_base});
  }
}

/// @overload
template <typename Index, typename Base>
auto tag_non_manifold_edge_fans(tf::non_manifold_edge_fans<Index> &fans,
                                Base &&base) {
  return tag_non_manifold_edge_fans(
      tf::make_non_manifold_edge_fans_like(tf::make_range(fans.edges),
                                           tf::make_range(fans.faces)),
      static_cast<Base &&>(base));
}

/// @overload
template <typename Index, typename Base>
auto tag_non_manifold_edge_fans(const tf::non_manifold_edge_fans<Index> &fans,
                                Base &&base) {
  return tag_non_manifold_edge_fans(
      tf::make_non_manifold_edge_fans_like(tf::make_range(fans.edges),
                                           tf::make_range(fans.faces)),
      static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_non_manifold_edge_fans(tf::non_manifold_edge_fans<Index> &&fans,
                                Base &&base) = delete;

namespace policy {
template <typename Policy> struct tag_non_manifold_edge_fans_op {
  Policy policy;
};

template <typename U, typename Policy>
auto operator|(U &&u, tag_non_manifold_edge_fans_op<Policy> t) {
  return tf::tag_non_manifold_edge_fans(
      tf::non_manifold_edge_fans_like<Policy>{std::move(t.policy)},
      static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup topology_policies
/// @brief Creates a pipe-able non-manifold edge fans tag operator.
///
/// Returns an object that can be used with pipe syntax to attach fans
/// to a range: `data | tf::tag_non_manifold_edge_fans(edges, faces)`.
template <typename EdgesIn, typename FacesIn>
auto tag_non_manifold_edge_fans(EdgesIn &&e, FacesIn &&f) {
  auto p = tf::topology::make_non_manifold_edge_fans_policy(
      static_cast<EdgesIn &&>(e), static_cast<FacesIn &&>(f));
  return policy::tag_non_manifold_edge_fans_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Index>
auto tag_non_manifold_edge_fans(tf::non_manifold_edge_fans<Index> &fans) {
  auto p = tf::topology::make_non_manifold_edge_fans_policy(
      tf::make_range(fans.edges), tf::make_range(fans.faces));
  return policy::tag_non_manifold_edge_fans_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Index>
auto tag_non_manifold_edge_fans(const tf::non_manifold_edge_fans<Index> &fans) {
  auto p = tf::topology::make_non_manifold_edge_fans_policy(
      tf::make_range(fans.edges), tf::make_range(fans.faces));
  return policy::tag_non_manifold_edge_fans_op<decltype(p)>{std::move(p)};
}

template <typename Index>
auto tag_non_manifold_edge_fans(tf::non_manifold_edge_fans<Index> &&fans) =
    delete;

/// @ingroup topology_policies
/// @brief Creates a pipe-able tag operator for non-manifold edge fans.
///
/// Generic overload of @ref tf::tag() that auto-detects the topology type.
template <typename Index>
auto tag(tf::non_manifold_edge_fans<Index> &fans) {
  auto p = tf::topology::make_non_manifold_edge_fans_policy(
      tf::make_range(fans.edges), tf::make_range(fans.faces));
  return policy::tag_non_manifold_edge_fans_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Index>
auto tag(const tf::non_manifold_edge_fans<Index> &fans) {
  auto p = tf::topology::make_non_manifold_edge_fans_policy(
      tf::make_range(fans.edges), tf::make_range(fans.faces));
  return policy::tag_non_manifold_edge_fans_op<decltype(p)>{std::move(p)};
}

template <typename Index>
auto tag(tf::non_manifold_edge_fans<Index> &&fans) = delete;

/// @overload
template <typename Policy>
auto tag(tf::non_manifold_edge_fans_like<Policy> &fans) {
  auto p = tf::topology::make_non_manifold_edge_fans_policy(
      tf::make_range(fans.edges), tf::make_range(fans.faces));
  return policy::tag_non_manifold_edge_fans_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Policy>
auto tag(const tf::non_manifold_edge_fans_like<Policy> &fans) {
  auto p = tf::topology::make_non_manifold_edge_fans_policy(
      tf::make_range(fans.edges), tf::make_range(fans.faces));
  return policy::tag_non_manifold_edge_fans_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Policy>
auto tag(tf::non_manifold_edge_fans_like<Policy> &&fans) {
  return tag(fans);
}

} // namespace tf

namespace std {
template <typename Policy, typename Base>
struct tuple_size<tf::policy::tag_non_manifold_edge_fans<Policy, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Policy, typename Base>
struct tuple_element<I, tf::policy::tag_non_manifold_edge_fans<Policy, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
