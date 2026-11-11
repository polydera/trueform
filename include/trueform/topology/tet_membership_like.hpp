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
/// @brief CRTP base for tet membership structures.
///
/// Provides the interface for structures that map vertices to the
/// tetrahedra containing them.
///
/// Use @ref tf::tet_membership for an owning container, or
/// @ref tf::make_tet_membership_like() to wrap an existing range.
///
/// @tparam Policy The underlying storage policy (e.g., @ref tf::offset_block_buffer).
template <typename Policy> struct tet_membership_like : Policy {
  tet_membership_like() = default;
  tet_membership_like(const Policy &policy) : Policy{policy} {}
  tet_membership_like(Policy &&policy) : Policy{std::move(policy)} {}
};

template <typename Policy>
auto unwrap(const tet_membership_like<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(tet_membership_like<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(tet_membership_like<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const tet_membership_like<Policy> &, T &&t) {
  return tet_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tet_membership_like<Policy> &, T &&t) {
  return tet_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tet_membership_like<Policy> &&, T &&t) {
  return tet_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup topology_connectivity
/// @brief Wrap a range as a tet membership view.
///
/// @tparam Range The underlying range type.
/// @param r The range to wrap.
/// @return A @ref tf::tet_membership_like view over the range.
template <typename Range> auto make_tet_membership_like(Range &&r) {
  return tf::tet_membership_like<std::decay_t<Range>>{static_cast<Range &&>(r)};
}
} // namespace tf
