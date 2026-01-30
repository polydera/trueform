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
#include "../core/buffer.hpp"

namespace tf {

/// @ingroup spatial_structures
/// @brief Index map for tree update operations.
///
/// Unlike `tf::index_map` where `kept_ids()` is the inverse of `f()`,
/// `tree_index_map` separates two concerns:
/// - `f()`: how all tree elements map to new indices (sentinel if removed)
/// - `dirty_ids()`: IDs of dirty/new elements to add to the delta tree
///
/// @tparam Range0 Range type for the forward mapping.
/// @tparam Range1 Range type for the dirty element IDs.
template <typename Range0, typename Range1>
class tree_index_map {
public:
  tree_index_map() = default;

  tree_index_map(const Range0 &_f, const Range1 &_dirty_ids)
      : _f{_f}, _dirty_ids{_dirty_ids} {}

  tree_index_map(Range0 &&_f, Range1 &&_dirty_ids)
      : _f{std::move(_f)}, _dirty_ids{std::move(_dirty_ids)} {}

  /// @brief Returns a mutable reference to the mapping function.
  auto f() -> Range0 & { return _f; }

  /// @brief Returns a const reference to the mapping function.
  auto f() const -> const Range0 & { return _f; }

  /// @brief Returns a mutable reference to the dirty IDs.
  auto dirty_ids() -> Range1 & { return _dirty_ids; }

  /// @brief Returns a const reference to the dirty IDs.
  auto dirty_ids() const -> const Range1 & { return _dirty_ids; }

private:
  Range0 _f;
  Range1 _dirty_ids;
};

template <typename Range0, typename Range1>
auto make_tree_index_map(Range0 &&_f, Range1 &&_dirty_ids) {
  return tree_index_map<std::decay_t<Range0>, std::decay_t<Range1>>{
      static_cast<Range0 &&>(_f), static_cast<Range1 &&>(_dirty_ids)};
}

template <typename Index>
class tree_index_map_buffer
    : public tree_index_map<tf::buffer<Index>, tf::buffer<Index>> {};

} // namespace tf
