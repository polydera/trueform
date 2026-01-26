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
#include <utility>

namespace tf {

/// @ingroup topology_connectivity
/// @brief CRTP base for manifold edge link structures.
///
/// Provides the interface for structures that store edge connectivity
/// in manifold meshes. For each directed edge, stores the peer face
/// that shares that edge (if any).
///
/// Use @ref tf::manifold_edge_link for an owning container, or
/// @ref tf::make_manifold_edge_link_like() to wrap an existing range.
///
/// @tparam Policy The underlying storage policy (e.g., @ref tf::blocked_buffer).
template <typename Policy> struct manifold_edge_link_like : Policy {
  manifold_edge_link_like() = default;
  manifold_edge_link_like(const Policy &policy) : Policy{policy} {}
  manifold_edge_link_like(Policy &&policy) : Policy{std::move(policy)} {}
};

template <typename Policy>
auto unwrap(const manifold_edge_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(manifold_edge_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(manifold_edge_link_like<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const manifold_edge_link_like<Policy> &, T &&t) {
  return manifold_edge_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(manifold_edge_link_like<Policy> &, T &&t) {
  return manifold_edge_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(manifold_edge_link_like<Policy> &&, T &&t) {
  return manifold_edge_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup topology_connectivity
/// @brief Wrap a range as a manifold edge link view.
///
/// @tparam Range The underlying range type.
/// @param r The range to wrap.
/// @return A @ref tf::manifold_edge_link_like view over the range.
template <typename Range> auto make_manifold_edge_link_like(Range &&r) {
  return tf::manifold_edge_link_like<std::decay_t<Range>>{
      static_cast<Range &&>(r)};
}
} // namespace tf
