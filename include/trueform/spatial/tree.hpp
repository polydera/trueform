/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/empty_aabb.hpp"
#include "./build_tree_nodes.hpp"
#include "./partitioning.hpp"
#include "./tree_config.hpp"

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
/// For details on the underlying methodology and performance characteristics,
/// refer to the accompanying technical paper.
///
/// @tparam Index The type used for primitive identifiers (typically an integer
/// type).
/// @tparam RealT The real-valued coordinate type (e.g., float or double).
/// @tparam N The spatial dimension (typically 2 or 3).

template <typename Index, typename RealT, std::size_t N> class tree {
public:
  tree() = default;

  template <typename Range, typename FC>
  tree(const Range &objects, const tf::tree_config<FC> &config) {
    build(objects, config);
  }

  template <typename Partitioner, typename Range, typename FC>
  auto build(const Range &objects, const tf::tree_config<FC> &config) -> void {
    _aabbs.allocate(objects.size());
    tf::parallel_apply(std::forward_as_tuple(objects, _aabbs),
                       [&](const auto &object, auto &aabb) {
                         aabb = config.make_aabb(object);
                       });
    tf::spatial::build_tree_nodes<Partitioner>(_nodes, _ids, _aabbs,
                                               config.node_config);
  }

  /// @brief Build the tree from a range of user-defined primitives.
  ///
  /// Constructs the tree by computing AABBs for the input range and recursively
  /// partitioning them into a balanced hierarchy, using the parameters defined
  /// in `config`.
  ///
  /// The input range must support indexed access, and `config` must define a
  /// bounding box construction function (`make_aabb`) that maps each object to
  /// an AABB.
  ///
  /// @tparam Range A random-access range of user-defined primitives.
  /// @tparam FC A callable type used by `tree_config` for bounding box
  /// generation.
  ///
  /// @param objects The range of primitives to insert into the tree.
  /// @param config Configuration parameters including the `make_aabb` function,
  ///               leaf size, and partitioning strategy. It is constructed via
  ///               @ref tf::config_tree free function for convenience.
  ///
  /// @return void
  template <typename Range, typename FC>
  auto build(const Range &objects, const tf::tree_config<FC> &config) -> void {
    return build<spatial::nth_element_t>(objects, config);
  }

  auto aabb() const {
    if (_nodes.size())
      return _nodes.front().aabb;
    else
      return tf::make_empty_aabb<RealT, N>();
  }

  /// @brief Access the axis-aligned bounding boxes (AABBs) of the input
  /// primitives.
  ///
  /// Returns the buffer of primitive-level AABBs computed during tree
  /// construction. These correspond one-to-one with the input range provided
  /// during `build(...)`.
  ///
  /// @return A constant reference to the internal AABB buffer.
  auto primitive_aabbs() const -> const tf::buffer<tf::aabb<RealT, N>> & {
    return _aabbs;
  }

  /// @brief Access the internal nodes of the tree.
  ///
  /// Returns the buffer of tree nodes representing the hierarchy. Each node
  /// stores its bounding box, child indices, and axis split information.
  ///
  /// @return A constant reference to the node buffer.
  auto nodes() const -> const tf::buffer<tf::tree_node<Index, RealT, N>> & {
    return _nodes;
  }

  /// @brief Access the leaf-level primitive ID buffer.
  ///
  /// Returns the buffer of primitive indices used during tree construction.
  /// Leaf nodes index into this buffer to refer to associated primitives.
  ///
  /// @return A constant reference to the primitive ID buffer.
  auto ids() const -> const tf::buffer<Index> & { return _ids; }

  /// @brief Clear all internal tree data.
  ///
  /// Empties the node, ID, and AABB buffers. This does not shrink capacity,
  /// but invalidates the current tree structure. A new build must be performed
  /// before queries can be issued again.
  auto clear() {
    _aabbs.clear();
    _nodes.clear();
    _ids.clear();
  }

private:
  tf::buffer<tf::aabb<RealT, N>> _aabbs;
  tf::buffer<tf::tree_node<Index, RealT, N>> _nodes;
  tf::buffer<Index> _ids;
};
} // namespace tf
