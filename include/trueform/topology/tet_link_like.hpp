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
/// @brief CRTP base for tet link structures.
///
/// Provides the interface for structures that store, per tetrahedron, the
/// tetrahedra sharing a facet with it.
///
/// Use @ref tf::tet_link for an owning container, or
/// @ref tf::make_tet_link_like() to wrap an existing range.
///
/// @tparam Policy The underlying storage policy (e.g., @ref tf::offset_block_buffer).
template <typename Policy> struct tet_link_like : Policy {
  tet_link_like() = default;
  tet_link_like(const Policy &policy) : Policy{policy} {}
  tet_link_like(Policy &&policy) : Policy{std::move(policy)} {}
};

template <typename Policy>
auto unwrap(const tet_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(tet_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(tet_link_like<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const tet_link_like<Policy> &, T &&t) {
  return tet_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tet_link_like<Policy> &, T &&t) {
  return tet_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tet_link_like<Policy> &&, T &&t) {
  return tet_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup topology_connectivity
/// @brief Wrap a range as a tet link view.
///
/// @tparam Range The underlying range type.
/// @param r The range to wrap.
/// @return A @ref tf::tet_link_like view over the range.
template <typename Range> auto make_tet_link_like(Range &&r) {
  return tf::tet_link_like<std::decay_t<Range>>{static_cast<Range &&>(r)};
}
} // namespace tf
