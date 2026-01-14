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
#include "../core/aabb_from.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "./partitioning.hpp"
#include "./tree/build_aabb_nodes.hpp"
#include "./tree/build_obb_nodes.hpp"
#include "./tree/build_obbrss_nodes.hpp"
#include "./tree_config.hpp"
#include "./tree_like.hpp"

namespace tf {

/// @ingroup spatial_structures
/// @brief A general-purpose spatial acceleration structure.
///
/// `tf::tree` is a balanced bounding volume hierarchy (BVH) optimized for fast
/// and flexible spatial queries, including intersection, self-intersection,
/// proximity queries, and nearest-neighbor searches.
///
/// Trees are constructed over user-defined primitives using parallel
/// partitioning strategies, ensuring efficient memory layout and predictable
/// query performance.
///
/// All query operations are exposed as free functions in the `tf::` namespace—
/// including @ref tf::search, @ref tf::search_self, and @ref
/// tf::nearness_search —and accept user-defined callbacks for full control over
/// traversal and primitive evaluation.
///
/// @tparam Index The type used for primitive identifiers (typically an integer
/// type).
/// @tparam BV The bounding volume type (e.g., `tf::aabb<RealT, N>`,
/// `tf::obb<RealT, N>`, or `tf::obbrss<RealT, N>`).
template <typename Index, typename BV>
class tree : public tree_like<spatial::tree_buffers<Index, BV>> {
  using base_t = tree_like<spatial::tree_buffers<Index, BV>>;

public:
  using base_t::ids;
  using base_t::nodes;
  using base_t::primitive_aabbs;

  using typename base_t::aabb_type;
  using typename base_t::bv_type;
  using typename base_t::coordinate_dims;
  using typename base_t::coordinate_type;
  using typename base_t::index_type;
  using typename base_t::node_type;

  tree() = default;

  /// @brief Construct tree from primitives with given configuration.
  ///
  /// @param primitives A range of primitives to build the tree from.
  /// @param config Configuration specifying inner and leaf node sizes.
  template <typename Range> tree(const Range &primitives, tree_config config) {
    build(primitives, config);
  }

  /// @brief Clear all internal tree data.
  auto clear() -> void {
    base_t::_primitive_aabbs.clear();
    base_t::_nodes.clear();
    base_t::_ids.clear();
  }

  /// @brief Build tree using a specific partitioning strategy.
  ///
  /// @tparam Partitioner The partitioning algorithm to use.
  /// @param primitives A range of primitives to build the tree from.
  /// @param config Configuration specifying inner and leaf node sizes.
  template <typename Partitioner, typename Range>
  auto build(const Range &primitives, tree_config config) -> void {
    base_t::_primitive_aabbs.allocate(primitives.size());
    tf::parallel_transform(
        primitives, base_t::_primitive_aabbs,
        [](const auto &x) { return tf::aabb_from(x); }, tf::checked);
    spatial::build_tree_nodes<Partitioner>(base_t::_nodes, base_t::_ids,
                                           primitives, base_t::_primitive_aabbs,
                                           config);
  }

  /// @brief Build tree from primitives using default partitioning.
  ///
  /// Uses nth_element partitioning strategy by default.
  ///
  /// @param primitives A range of primitives to build the tree from.
  /// @param config Configuration specifying inner and leaf node sizes.
  template <typename Range>
  auto build(const Range &primitives, tree_config config) -> void {
    build<spatial::nth_element_t>(primitives, config);
  }
};

} // namespace tf
