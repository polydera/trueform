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
#include <array>
#include <cstdint>
namespace tf::spatial {
template <typename Index, typename BV> struct tree_node {
  /// Special marker for an empty node.
  static constexpr Index empty_tag = -2;

  /// Special marker for a leaf node.
  static constexpr Index leaf_tag = -1;

  BV bv;

  std::array<Index, 2> _data;

  /// @brief Indicates the splitting axis for inner nodes, or a special tag for
  /// leaf/empty.
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

  /// @brief Set the node’s data payload (e.g., child indices or primitive
  /// range).
  /// @param first First element (e.g., child index or range start).
  /// @param second Second element (e.g., child index or range end).
  auto set_data(Index first, Index second) {
    _data[0] = first, _data[1] = second;
  }

  /// @brief Get the node's data array.
  /// @return A const reference to the internal data array.
  auto get_data() const -> const std::array<Index, 2> { return _data; }
};
} // namespace tf::spatial
