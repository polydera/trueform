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
#include "./components/policy.hpp"
#include <utility>

namespace tf {

/// @ingroup topology_components
/// @brief CRTP base for connected component labels structures.
///
/// The underlying `Policy` carries the per-element labels container plus
/// the component count and connectivity rule used to produce them
/// (see @ref tf::topology::connected_component_labels_policy).
///
/// Use @ref tf::connected_component_labels for an owning container, or
/// @ref tf::make_connected_component_labels_like() to wrap an existing
/// labels range with explicit scalars.
///
/// @tparam Policy The underlying storage+scalars policy.
template <typename Policy>
struct connected_component_labels_like : Policy {
  using Policy::Policy;
  connected_component_labels_like() = default;
  connected_component_labels_like(const Policy &p) : Policy{p} {}
  connected_component_labels_like(Policy &&p) : Policy{std::move(p)} {}

  using typename Policy::label_type;
};

template <typename Policy>
auto unwrap(const connected_component_labels_like<Policy> &c)
    -> decltype(auto) {
  return static_cast<const Policy &>(c);
}

template <typename Policy>
auto unwrap(connected_component_labels_like<Policy> &c) -> decltype(auto) {
  return static_cast<Policy &>(c);
}

template <typename Policy>
auto unwrap(connected_component_labels_like<Policy> &&c) -> decltype(auto) {
  return static_cast<Policy &&>(c);
}

template <typename Policy, typename T>
auto wrap_like(const connected_component_labels_like<Policy> &, T &&t) {
  return connected_component_labels_like<std::decay_t<T>>{
      static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(connected_component_labels_like<Policy> &, T &&t) {
  return connected_component_labels_like<std::decay_t<T>>{
      static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(connected_component_labels_like<Policy> &&, T &&t) {
  return connected_component_labels_like<std::decay_t<T>>{
      static_cast<T &&>(t)};
}

/// @ingroup topology_components
/// @brief Wrap an external labels range as a view-typed connected
/// component labels structure.
///
/// @tparam Range The labels container type to wrap (deduced).
/// @tparam LabelType The integer type for component labels (deduced from `n_components`).
/// @param r The labels container.
/// @param n_components The total number of distinct components.
template <typename Range, typename LabelType>
auto make_connected_component_labels_like(Range &&r, LabelType n_components) {
  auto policy = tf::topology::make_connected_component_labels_policy(
      static_cast<Range &&>(r), n_components);
  return connected_component_labels_like<decltype(policy)>{std::move(policy)};
}

} // namespace tf
