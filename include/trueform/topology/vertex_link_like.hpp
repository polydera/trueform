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
/// @brief CRTP base for vertex link structures.
///
/// Provides the interface for structures that store the 1-ring neighborhood
/// of each vertex. The underlying `Policy` typically stores per-vertex lists
/// of neighboring vertex or face indices.
///
/// Use @ref tf::vertex_link for an owning container, or
/// @ref tf::make_vertex_link_like() to wrap an existing range.
///
/// @tparam Policy The underlying storage policy (e.g., @ref tf::offset_block_buffer).
template <typename Policy> struct vertex_link_like : Policy {
  vertex_link_like() = default;
  vertex_link_like(const Policy &policy) : Policy{policy} {}
  vertex_link_like(Policy &&policy) : Policy{std::move(policy)} {}
};

template <typename Policy>
auto unwrap(const vertex_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(vertex_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(vertex_link_like<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const vertex_link_like<Policy> &, T &&t) {
  return vertex_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(vertex_link_like<Policy> &, T &&t) {
  return vertex_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(vertex_link_like<Policy> &&, T &&t) {
  return vertex_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup topology_connectivity
/// @brief Wrap a range as a vertex link view.
///
/// @tparam Range The underlying range type.
/// @param r The range to wrap.
/// @return A @ref tf::vertex_link_like view over the range.
template <typename Range> auto make_vertex_link_like(Range &&r) {
  return tf::vertex_link_like<std::decay_t<Range>>{
      static_cast<Range &&>(r)};
}
} // namespace tf

