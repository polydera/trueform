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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/buffer.hpp"
#include "../core/index_map.hpp"
#include "../core/views/indirect_range.hpp"
#include "./mod_tree_like.hpp"
#include "./partitioning.hpp"
#include "./tree/build_aabb_nodes.hpp"
#include "./tree_config.hpp"
#include <algorithm>

namespace tf::spatial {

template <typename Index, typename BV>
auto clear_tree_buffers(tree_buffers<Index, BV> &buffers) -> void {
  buffers.primitive_aabbs_buffer().clear();
  buffers.nodes_buffer().clear();
  buffers.ids_buffer().clear();
}

// Clear tree_buffers_core (nodes + ids only)
template <typename Index, typename BV>
auto clear_tree_buffers_core(tree_buffers_core<Index, BV> &buffers) -> void {
  buffers.nodes_buffer().clear();
  buffers.ids_buffer().clear();
}

template <typename Partitioner, typename Index, typename BV, typename Range>
auto build_tree_buffers(tree_buffers<Index, BV> &buffers,
                        const Range &primitives, tree_config config,
                        bool use_ids = false) -> void {
  if (!use_ids) {
    buffers.primitive_aabbs_buffer().allocate(primitives.size());
    tf::parallel_transform(
        primitives, buffers.primitive_aabbs_buffer(),
        [](const auto &x) { return tf::aabb_from(x); }, tf::checked);
  }
  build_tree_nodes<Partitioner>(buffers.nodes_buffer(), buffers.ids_buffer(),
                                primitives, buffers.primitive_aabbs_buffer(),
                                config, use_ids);
}

// Build tree nodes for tree_buffers_core using external aabbs
template <typename Partitioner, typename Index, typename BV, typename Range,
          typename AabbRange>
auto build_tree_nodes_with_aabbs(tree_buffers_core<Index, BV> &buffers,
                                 const Range &primitives,
                                 const AabbRange &aabbs, tree_config config,
                                 bool use_ids = false) -> void {
  build_tree_nodes<Partitioner>(buffers.nodes_buffer(), buffers.ids_buffer(),
                                primitives, aabbs, config, use_ids);
}

} // namespace tf::spatial

namespace tf {
/**
 * @ingroup spatial_structures
 * @brief A dynamic spatial tree that combines a persistent main tree with a
 * transient delta tree.
 *
 * This structure supports efficient incremental updates by separating static
 * data (stored in the main @ref tf::tree) from newly added or moved data
 * (stored in the delta @ref tf::tree). The delta tree is rebuilt from scratch
 * on each update, while the main tree is pruned.
 *
 * @tparam Index Index type used for object references
 * @tparam BV    Bounding volume type (e.g., tf::aabb, tf::obb, tf::obbrss)
 *
 * @see tf::tree
 * @see tf::index_map
 */
template <typename Index, typename BV>
class mod_tree : public mod_tree_like<spatial::mod_tree_buffers<Index, BV>> {
  using base_t = mod_tree_like<spatial::mod_tree_buffers<Index, BV>>;

public:
  using base_t::delta_ids;
  using base_t::delta_ids_buffer;
  using base_t::delta_tree;
  using base_t::delta_tree_buffer;
  using base_t::main_tree;
  using base_t::main_tree_buffer;

  using typename base_t::aabb_type;
  using typename base_t::bv_type;
  using typename base_t::coordinate_dims;
  using typename base_t::coordinate_type;
  using typename base_t::index_type;
  using typename base_t::node_type;

  mod_tree() = default;
  /**
   * @brief Builds the main tree from a given range of objects using a specified
   * partitioner.
   *
   * This clears any delta data and constructs the main tree.
   *
   * @tparam Partitioner A strategy type used for partitioning during tree
   * construction
   * @tparam Range       A range of geometric objects
   * @param objects      The input range of geometric objects
   * @param config       Tree configuration (see ref::tree_config)
   */
  template <typename Partitioner, typename Range>
  auto build(const Range &objects, tree_config config) -> void {
    delta_ids_buffer().clear();
    spatial::clear_tree_buffers_core(delta_tree_buffer());
    // Allocate shared primitive_aabbs and compute AABBs
    base_t::primitive_aabbs_buffer().allocate(objects.size());
    tf::parallel_transform(
        objects, base_t::primitive_aabbs_buffer(),
        [](const auto &x) { return tf::aabb_from(x); }, tf::checked);
    // Build main tree nodes using the shared aabbs
    spatial::build_tree_nodes_with_aabbs<Partitioner>(
        main_tree_buffer(), objects, base_t::primitive_aabbs_buffer(), config);
  }

  /**
   * @brief Builds the main tree using the default nth-element partitioner.
   *
   * Convenience overload of @ref ::build with a default partitioner.
   *
   * @tparam Range A range of geometric objects
   * @param objects The input range of geometric objects
   * @param config  Tree configuration (see ref::tree_config)
   */
  template <typename Range>
  auto build(const Range &objects, tree_config config) -> void {
    return build<spatial::nth_element_t>(objects, config);
  }

  /**
   * @brief Updates the tree with new or modified objects.
   *
   * This prunes the main tree using the `keep_if` predicate and constructs the
   * delta tree from the given set of new objects and their corresponding IDs.
   *
   * @tparam Range0  A range of geometric objects
   * @tparam Range1  A range of object indices
   * @tparam F       A unary predicate defining which indices to keep
   * @param objects  New or updated geometric objects
   * @param ids      Indices of the updated objects
   * @param keep_if  Predicate returning true for IDs that should remain in the
   * tree
   * @param config   Tree configuration (see ref::tree_config)
   */
  template <typename Range0, typename Range1, typename F>
  auto update(const Range0 &objects, const Range1 &ids, const F &keep_if,
              tree_config config) {
    // Estimate new delta size
    Index estimated_delta = Index(delta_ids_buffer().size() + ids.size());
    if (estimated_delta * 2 > Index(main_tree_buffer().ids().size())) {
      build(objects, config);
      return;
    }
    update_main_tree(keep_if);
    update_delta_tree(objects, ids, keep_if, config);
  }

  /**
   * @brief Updates the tree using an index reindex_map (via ref::index_map).
   *
   * Used when object IDs have been remapped or reordered. This prunes the main
   * tree and constructs a new delta tree using the remapped IDs.
   *
   * @tparam Range     A range of geometric objects
   * @tparam Range1    Underlying type for `index_map::f()` (forward map)
   * @tparam Range2    Underlying type for `index_map::kept_ids()` (kept/valid
   * IDs)
   * @tparam F         A predicate that determines which IDs to keep
   * @param objects    New or updated geometric objects
   * @param index_map    Index reindex_map (see ref::index_map)
   * @param keep_if    Predicate for keeping existing IDs
   * @param config     Tree configuration (see ref::tree_config)
   */
  template <typename Range, typename Range1, typename Range2, typename F>
  auto update_tree(const Range &objects,
                   const tf::index_map<Range1, Range2> &index_map,
                   const F &keep_if, tree_config config) {
    // Estimate new delta size
    Index estimated_delta =
        Index(delta_ids_buffer().size() + index_map.kept_ids().size());
    if (estimated_delta * 2 > Index(main_tree_buffer().ids().size())) {
      build(objects, config);
      return;
    }
    update_main_tree(objects, index_map.f(), keep_if);
    update_delta_tree(objects, index_map, keep_if, config);
  }

  /**
   * @brief Clears all data from both the main and delta trees.
   */
  auto clear() -> void {
    base_t::primitive_aabbs_buffer().clear();
    spatial::clear_tree_buffers_core(main_tree_buffer());
    spatial::clear_tree_buffers_core(delta_tree_buffer());
    delta_ids_buffer().clear();
  }

private:
  template <typename Objects, typename Range, typename F>
  auto update_main_tree(const Objects &objects, const Range &id_map,
                        const F &keep_if) {
    auto &&ids = main_tree_buffer().ids();
    // Allocate shared primitive_aabbs buffer and recompute from objects
    base_t::primitive_aabbs_buffer().allocate(objects.size());
    tf::parallel_transform(
        objects, base_t::primitive_aabbs_buffer(),
        [](const auto &x) { return tf::aabb_from(x); }, tf::checked);

    // Remap IDs and partition
    tf::parallel_for_each(main_tree_buffer().nodes(), [&](auto &node) {
      if (!node.is_leaf())
        return;
      auto &&data = node.get_data();
      auto &&c_ids = tf::make_range(ids.begin() + data[0], data[1]);
      for (auto &x : c_ids)
        x = id_map[x];
      node.set_data(data[0],
                    std::partition(c_ids.begin(), c_ids.end(), keep_if) -
                        c_ids.begin());
    });
  }
  template <typename F> auto update_main_tree(const F &keep_if) {
    auto &&ids = main_tree_buffer().ids();
    tf::parallel_for_each(main_tree_buffer().nodes(), [&](auto &node) {
      if (!node.is_leaf())
        return;
      auto &&data = node.get_data();
      auto &&c_ids = tf::make_range(ids.begin() + data[0], data[1]);
      // we move invalidated ids to the end of the subrange
      // and set new size
      node.set_data(data[0],
                    std::partition(c_ids.begin(), c_ids.end(), keep_if) -
                        c_ids.begin());
    });
  }

  template <typename Range0, typename Range1, typename F>
  auto update_delta_tree(const Range0 &objects, const Range1 &ids,
                         const F &keep_if, tree_config config) {
    auto n_additional_objects = ids.size();
    delta_ids_buffer().allocate(n_additional_objects +
                                delta_ids_buffer().size());
    // keep all old ids that are not in the
    // other region (as those will get copied from there)
    auto &&old_delta_ids = delta_tree_buffer().ids();
    auto write_to = std::copy_if(old_delta_ids.begin(), old_delta_ids.end(),
                                 delta_ids_buffer().begin(), keep_if);
    write_to = std::copy(ids.begin(), ids.end(), write_to);
    // this will only bump down the end pointer
    delta_ids_buffer().allocate(write_to - delta_ids_buffer().begin());

    // Pre-populate tree ids with global ids so tree can use shared aabbs
    delta_tree_buffer().ids_buffer().allocate(delta_ids_buffer().size());
    tf::parallel_copy(delta_ids_buffer(), delta_tree_buffer().ids_buffer());

    // Build delta tree using the shared primitive_aabbs and pre-set global ids
    spatial::build_tree_nodes_with_aabbs<spatial::nth_element_t>(
        delta_tree_buffer(),
        tf::make_indirect_range(delta_ids_buffer(), objects),
        base_t::primitive_aabbs_buffer(), config, /*use_ids=*/true);
  }

  template <typename Range, typename Range1, typename Range2, typename F>
  auto update_delta_tree(const Range &objects,
                         const tf::index_map<Range1, Range2> &index_map,
                         const F &keep_if, tree_config config) {
    auto n_additional_objects = index_map.kept_ids().size();
    delta_ids_buffer().allocate(n_additional_objects +
                                delta_ids_buffer().size());
    // keep all old ids that are not in the
    // other region (as those will get copied from there)
    auto &&old_delta_ids = delta_tree_buffer().ids();
    auto &&mapped_ids = tf::make_indirect_range(old_delta_ids, index_map.f());
    auto write_to = std::copy_if(mapped_ids.begin(), mapped_ids.end(),
                                 delta_ids_buffer().begin(), keep_if);
    write_to = std::copy(index_map.kept_ids().begin(),
                         index_map.kept_ids().end(), write_to);
    // this will only bump down the end pointer
    delta_ids_buffer().allocate(write_to - delta_ids_buffer().begin());

    // Pre-populate tree ids with global ids so tree can use shared aabbs
    delta_tree_buffer().ids_buffer().allocate(delta_ids_buffer().size());
    tf::parallel_copy(delta_ids_buffer(), delta_tree_buffer().ids_buffer());

    // Build delta tree using the shared primitive_aabbs and pre-set global ids
    // (aabbs were already recomputed by update_main_tree)
    spatial::build_tree_nodes_with_aabbs<spatial::nth_element_t>(
        delta_tree_buffer(),
        tf::make_indirect_range(delta_ids_buffer(), objects),
        base_t::primitive_aabbs_buffer(), config, /*use_ids=*/true);
  }
};
} // namespace tf
