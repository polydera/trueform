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
#include "../connected_component_labels.hpp"
#include <utility>

namespace tf {
namespace policy {

template <typename Policy, typename Base>
struct tag_connected_component_labels;

template <typename Policy, typename Base>
auto has_connected_component_labels(
    const tag_connected_component_labels<Policy, Base> *) -> std::true_type;

auto has_connected_component_labels(const void *) -> std::false_type;

} // namespace policy

/// @ingroup topology_policies
/// @brief Checks if a type has connected component labels policy attached.
///
/// True if the type was wrapped with @ref tf::tag_connected_component_labels().
///
/// @tparam T The type to check.
template <typename T>
inline constexpr bool has_connected_component_labels_policy =
    decltype(policy::has_connected_component_labels(
        static_cast<const std::decay_t<T> *>(nullptr)))::value;

namespace policy {
template <typename Policy, typename Base>
struct tag_connected_component_labels : Base {
  using Base::operator=;

  tag_connected_component_labels(
      tf::connected_component_labels_like<Policy> _cl, const Base &base)
      : Base{base}, _cl{std::move(_cl)} {}

  tag_connected_component_labels(
      tf::connected_component_labels_like<Policy> _cl, Base &&base)
      : Base{std::move(base)}, _cl{std::move(_cl)} {}

  auto connected_component_labels() const
      -> const tf::connected_component_labels_like<Policy> & {
    return _cl;
  }

  auto connected_component_labels()
      -> tf::connected_component_labels_like<Policy> & {
    return _cl;
  }

private:
  tf::connected_component_labels_like<Policy> _cl;

  friend auto unwrap(const tag_connected_component_labels &val) -> const Base & {
    return static_cast<const Base &>(val);
  }

  friend auto unwrap(tag_connected_component_labels &val) -> Base & {
    return static_cast<Base &>(val);
  }

  friend auto unwrap(tag_connected_component_labels &&val) -> Base && {
    return static_cast<Base &&>(val);
  }

  template <typename T>
  friend auto wrap_like(const tag_connected_component_labels &val, T &&t) {
    return tag_connected_component_labels<Policy, std::decay_t<T>>{
        val._cl, static_cast<T &&>(t)};
  }
};
} // namespace policy

template <typename Policy, typename Base>
struct static_size<policy::tag_connected_component_labels<Policy, Base>>
    : static_size<Base> {};

/// @ingroup topology_policies
/// @brief Attaches connected component labels to a base type.
///
/// Creates a wrapper that carries the labels view alongside the base
/// data. The result provides a `.connected_component_labels()` accessor.
/// Use with pipe syntax: `data | tf::tag_connected_component_labels(cl)`.
///
/// @tparam Policy The connected component labels policy type.
/// @tparam Base The base type to wrap.
template <typename Policy, typename Base>
auto tag_connected_component_labels(
    tf::connected_component_labels_like<Policy> &&_cl, Base &&base) {
  if constexpr (has_connected_component_labels_policy<Base>)
    if constexpr (std::is_rvalue_reference_v<Base &&>)
      return static_cast<Base>(base);
    else
      return static_cast<Base &&>(base);
  else {
    auto &b_base = unwrap(base);
    return wrap_like(
        base,
        policy::tag_connected_component_labels<Policy,
                                               std::decay_t<decltype(b_base)>>{
            std::move(_cl), b_base});
  }
}

/// @overload
template <typename Index, typename Base>
auto tag_connected_component_labels(tf::connected_component_labels<Index> &cl,
                                    Base &&base) {
  return tag_connected_component_labels(
      tf::make_connected_component_labels_like(tf::make_range(cl.labels),
                                               cl.n_components),
      static_cast<Base &&>(base));
}

/// @overload
template <typename Index, typename Base>
auto tag_connected_component_labels(
    const tf::connected_component_labels<Index> &cl, Base &&base) {
  return tag_connected_component_labels(
      tf::make_connected_component_labels_like(tf::make_range(cl.labels),
                                               cl.n_components),
      static_cast<Base &&>(base));
}

template <typename Index, typename Base>
auto tag_connected_component_labels(tf::connected_component_labels<Index> &&cl,
                                    Base &&base) = delete;

namespace policy {
template <typename Policy> struct tag_connected_component_labels_op {
  Policy policy;
};

template <typename U, typename Policy>
auto operator|(U &&u, tag_connected_component_labels_op<Policy> t) {
  return tf::tag_connected_component_labels(
      tf::connected_component_labels_like<Policy>{std::move(t.policy)},
      static_cast<U &&>(u));
}
} // namespace policy

/// @ingroup topology_policies
/// @brief Creates a pipe-able connected component labels tag operator.
///
/// Returns an object that can be used with pipe syntax to attach
/// labels to a range: `data | tf::tag_connected_component_labels(cl)`.
template <typename Range, typename LabelType>
auto tag_connected_component_labels(Range &&r, LabelType n_components) {
  auto p = tf::topology::make_connected_component_labels_policy(
      static_cast<Range &&>(r), n_components);
  return policy::tag_connected_component_labels_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Index>
auto tag_connected_component_labels(tf::connected_component_labels<Index> &cl) {
  auto p = tf::topology::make_connected_component_labels_policy(
      tf::make_range(cl.labels), cl.n_components);
  return policy::tag_connected_component_labels_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Index>
auto tag_connected_component_labels(
    const tf::connected_component_labels<Index> &cl) {
  auto p = tf::topology::make_connected_component_labels_policy(
      tf::make_range(cl.labels), cl.n_components);
  return policy::tag_connected_component_labels_op<decltype(p)>{std::move(p)};
}

template <typename Index>
auto tag_connected_component_labels(
    tf::connected_component_labels<Index> &&cl) = delete;

/// @ingroup topology_policies
/// @brief Creates a pipe-able tag operator for connected component labels.
///
/// Generic overload of @ref tf::tag() that auto-detects the topology type.
template <typename Index>
auto tag(tf::connected_component_labels<Index> &cl) {
  auto p = tf::topology::make_connected_component_labels_policy(
      tf::make_range(cl.labels), cl.n_components);
  return policy::tag_connected_component_labels_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Index>
auto tag(const tf::connected_component_labels<Index> &cl) {
  auto p = tf::topology::make_connected_component_labels_policy(
      tf::make_range(cl.labels), cl.n_components);
  return policy::tag_connected_component_labels_op<decltype(p)>{std::move(p)};
}

template <typename Index>
auto tag(tf::connected_component_labels<Index> &&cl) = delete;

/// @overload
template <typename Policy>
auto tag(tf::connected_component_labels_like<Policy> &cl) {
  auto p = tf::topology::make_connected_component_labels_policy(
      tf::make_range(cl.labels), cl.n_components);
  return policy::tag_connected_component_labels_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Policy>
auto tag(const tf::connected_component_labels_like<Policy> &cl) {
  auto p = tf::topology::make_connected_component_labels_policy(
      tf::make_range(cl.labels), cl.n_components);
  return policy::tag_connected_component_labels_op<decltype(p)>{std::move(p)};
}

/// @overload
template <typename Policy>
auto tag(tf::connected_component_labels_like<Policy> &&cl) {
  return tag(cl);
}

} // namespace tf

namespace std {
template <typename Policy, typename Base>
struct tuple_size<tf::policy::tag_connected_component_labels<Policy, Base>>
    : tuple_size<Base> {};

template <std::size_t I, typename Policy, typename Base>
struct tuple_element<I,
                     tf::policy::tag_connected_component_labels<Policy, Base>> {
  using type = typename std::iterator_traits<
      decltype(declval<Base>().begin())>::value_type;
};
} // namespace std
