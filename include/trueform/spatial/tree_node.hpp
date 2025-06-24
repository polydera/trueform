/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/aabb.hpp"
#include <cstdint>

namespace tf {

/// @ingroup spatial_structures
/// @brief A node in the spatial tree hierarchy.
///
/// Each node contains an axis-aligned bounding box (AABB), data indices, and an axis indicator.
/// The node can represent either an internal node, a leaf, or an empty node.
/// Leaf and empty status are encoded via special `axis` tag values.
///
/// @tparam Index The type used for indexing child nodes or primitive references.
/// @tparam RealType The floating-point type used for spatial coordinates.
/// @tparam N The spatial dimension (e.g., 2 for 2D, 3 for 3D).
template <typename Index, typename RealType, std::size_t N> 
struct tree_node {
  /// Special marker for an empty node.
  static constexpr Index empty_tag = -2;
  
  /// Special marker for a leaf node.
  static constexpr Index leaf_tag = -1;

  /// @brief Bounding box of the node.
  tf::aabb<RealType, N> aabb;

  /// @brief Encodes either child node indices or primitive ID range, depending on node type.
  ///
  /// The relevant range starts at `_data[0]` and contains `_data[1]` elements.
  std::array<Index, 2> _data;

  /// @brief Indicates the splitting axis for inner nodes, or a special tag for leaf/empty.
  std::int8_t axis;

  /// @brief Set the splitting axis.
  /// @param _axis The axis along which the node was split.
  auto set_axis(std::int8_t _axis) { axis = _axis; }

  /// @brief Mark the node as a leaf.
  auto set_as_leaf() { axis = leaf_tag; }

  /// @brief Mark the node as empty.
  auto set_as_empty() { axis = empty_tag; }

  /// @brief Check if the node is a leaf.
  /// @return True if the node is marked as a leaf.
  auto is_leaf() const { return axis == leaf_tag; }

  /// @brief Check if the node is empty.
  /// @return True if the node is marked as empty.
  auto is_empty() const { return axis == empty_tag; }

  /// @brief Set the node’s data payload (e.g., child indices or primitive range).
  /// @param first First element (e.g., child index or range start).
  /// @param second Second element (e.g., child index or range end).
  auto set_data(Index first, Index second) {
    _data[0] = first, _data[1] = second;
  }

  /// @brief Get the node's data array.
  /// @return A const reference to the internal data array.
  auto get_data() const -> const std::array<Index, 2> { return _data; }
};

} // namespace tf
