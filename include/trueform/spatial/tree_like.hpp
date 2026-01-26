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
#include "./tree/tree_buffers.hpp"
#include "./tree/tree_ranges.hpp"
#include <utility>

namespace tf {

/// @ingroup spatial_structures
/// @brief CRTP base class for static spatial trees.
///
/// `tree_like` provides the common interface for all spatial tree types,
/// exposing access to nodes, primitive IDs, and bounding volumes. Concrete
/// trees like @ref tf::tree inherit from this class.
///
/// @tparam Policy The underlying storage policy providing nodes, ids, and aabbs.
template <typename Policy> struct tree_like : Policy {
  tree_like() = default;
  tree_like(const Policy &policy) : Policy{policy} {}
  tree_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::ids;
  using Policy::nodes;
  using Policy::primitive_aabbs;

  using typename Policy::aabb_type;
  using typename Policy::bv_type;
  using typename Policy::coordinate_dims;
  using typename Policy::coordinate_type;
  using typename Policy::index_type;
  using typename Policy::node_type;

  /// @brief Get the root bounding volume of the tree.
  auto bv() const -> const bv_type & { return nodes()[0].bv; }
};

template <typename Policy>
auto unwrap(const tree_like<Policy> &t) -> decltype(auto) {
  return static_cast<const Policy &>(t);
}

template <typename Policy>
auto unwrap(tree_like<Policy> &t) -> decltype(auto) {
  return static_cast<Policy &>(t);
}

template <typename Policy>
auto unwrap(tree_like<Policy> &&t) -> decltype(auto) {
  return static_cast<Policy &&>(t);
}

template <typename Policy, typename T>
auto wrap_like(const tree_like<Policy> &, T &&t) {
  return tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tree_like<Policy> &, T &&t) {
  return tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tree_like<Policy> &&, T &&t) {
  return tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

/// @ingroup spatial_structures
/// @brief Wrap a range as a tree_like.
///
/// @tparam Range The range type to wrap.
/// @param r The range to wrap.
/// @return A tree_like wrapping the range.
template <typename Range> auto make_tree_like(Range &&r) {
  return tree_like<std::decay_t<Range>>{static_cast<Range &&>(r)};
}

/// @ingroup spatial_structures
/// @brief Create a lightweight view of a tree.
///
/// The returned view references the original tree's data without copying.
/// The original tree must outlive the view.
///
/// @tparam Policy The tree's policy type.
/// @param t The tree to create a view of.
/// @return A tree_like view referencing the original data.
template <typename Policy>
auto make_tree_view(const tree_like<Policy> &t) {
  using index_type = typename Policy::index_type;
  using bv_type = typename Policy::bv_type;
  return tree_like<spatial::tree_ranges<index_type, bv_type>>{
      t.nodes(), t.ids(), t.primitive_aabbs()};
}

} // namespace tf
