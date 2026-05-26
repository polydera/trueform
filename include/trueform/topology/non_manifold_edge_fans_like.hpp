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
#include "./non_manifold_edge_fans/policy.hpp"
#include <utility>

namespace tf {

/// @ingroup topology_components
/// @brief CRTP base for non-manifold edge fan structures.
///
/// The underlying `Policy` carries two parallel containers: per-edge
/// endpoint pairs and per-edge incident face blocks
/// (see @ref tf::topology::non_manifold_edge_fans_policy).
///
/// Use @ref tf::non_manifold_edge_fans for an owning container, or
/// @ref tf::make_non_manifold_edge_fans_like() to wrap existing ranges.
///
/// @tparam Policy The underlying storage policy.
template <typename Policy>
struct non_manifold_edge_fans_like : Policy {
  using Policy::Policy;
  non_manifold_edge_fans_like() = default;
  non_manifold_edge_fans_like(const Policy &p) : Policy{p} {}
  non_manifold_edge_fans_like(Policy &&p) : Policy{std::move(p)} {}
};

template <typename Policy>
auto unwrap(const non_manifold_edge_fans_like<Policy> &c) -> decltype(auto) {
  return static_cast<const Policy &>(c);
}

template <typename Policy>
auto unwrap(non_manifold_edge_fans_like<Policy> &c) -> decltype(auto) {
  return static_cast<Policy &>(c);
}

template <typename Policy>
auto unwrap(non_manifold_edge_fans_like<Policy> &&c) -> decltype(auto) {
  return static_cast<Policy &&>(c);
}

template <typename Policy, typename T>
auto wrap_like(const non_manifold_edge_fans_like<Policy> &, T &&t) {
  return non_manifold_edge_fans_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(non_manifold_edge_fans_like<Policy> &, T &&t) {
  return non_manifold_edge_fans_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(non_manifold_edge_fans_like<Policy> &&, T &&t) {
  return non_manifold_edge_fans_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup topology_components
/// @brief Wrap external edges + faces ranges as a view-typed
///        non-manifold edge fans structure.
///
/// @tparam EdgesIn The endpoint-pairs container type (deduced).
/// @tparam FacesIn The per-edge face-blocks container type (deduced).
/// @param e The endpoint pairs.
/// @param f The per-edge face blocks.
template <typename EdgesIn, typename FacesIn>
auto make_non_manifold_edge_fans_like(EdgesIn &&e, FacesIn &&f) {
  auto policy = tf::topology::make_non_manifold_edge_fans_policy(
      static_cast<EdgesIn &&>(e), static_cast<FacesIn &&>(f));
  return non_manifold_edge_fans_like<decltype(policy)>{std::move(policy)};
}

} // namespace tf
