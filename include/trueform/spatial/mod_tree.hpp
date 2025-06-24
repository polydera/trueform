/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/buffer.hpp"
#include "../core/index_map.hpp"
#include "../core/views/indirect_range.hpp"
#include "./tree.hpp"
#include <algorithm>

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
 * @tparam Index    Index type used for object references
 * @tparam RealType Floating-point type used for coordinates
 * @tparam Dims     Dimensionality of the space (e.g., 2 or 3)
 *
 * @see tf::tree
 * @see tf::index_map
 */
template <typename Index, typename RealType, std::size_t Dims> class mod_tree {
public:
  /**
   * @brief Builds the main tree from a given range of objects using a specified
   * partitioner.
   *
   * This clears any delta data and constructs the main tree.
   *
   * @tparam Partitioner A strategy type used for partitioning during tree
   * construction
   * @tparam Range       A range of geometric objects
   * @tparam FC          A callable or policy object passed to the tree config
   * @param objects      The input range of geometric objects
   * @param config       Tree configuration (see ref::tree_config)
   */
  template <typename Partitioner, typename Range, typename FC>
  auto build(const Range &objects, const tf::tree_config<FC> &config) -> void {
    _delta_ids.clear();
    _delta_tree.clear();
    _main_tree.build(objects, make_aabb, config);
  }

  /**
   * @brief Builds the main tree using the default nth-element partitioner.
   *
   * Convenience overload of @ref ::build with a default partitioner.
   *
   * @tparam Range A range of geometric objects
   * @tparam FC    A callable or policy object passed to the tree config
   * @param objects The input range of geometric objects
   * @param config  Tree configuration (see ref::tree_config)
   */
  template <typename Range, typename FC>
  auto build(const Range &objects, const tf::tree_config<FC> &config) -> void {
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
   * @tparam FC      A callable or policy object passed to the tree config
   * @param objects  New or updated geometric objects
   * @param ids      Indices of the updated objects
   * @param keep_if  Predicate returning true for IDs that should remain in the
   * tree
   * @param config   Tree configuration (see ref::tree_config)
   */
  template <typename Range0, typename Range1, typename F, typename FC>
  auto update(const Range0 &objects, const Range1 &ids, const F &keep_if,
              const tf::tree_config<FC> &config) {
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
   * @tparam FC        A callable or policy object passed to the tree config
   * @param objects    New or updated geometric objects
   * @param index_map    Index reindex_map (see ref::index_map)
   * @param keep_if    Predicate for keeping existing IDs
   * @param config     Tree configuration (see ref::tree_config)
   */
  template <typename Range, typename Range1, typename Range2, typename F,
            typename FC>
  auto update_tree(const Range &objects,
                   const tf::index_map<Range1, Range2> &index_map,
                   const F &keep_if, const tf::tree_config<FC> &config) {
    update_main_tree(index_map.f(), keep_if);
    update_delta_tree(objects, index_map, keep_if, config);
  }

  /**
   * @brief Returns a const reference to the main tree.
   */
  auto main_tree() const -> const tree<Index, RealType, Dims> & {
    return _main_tree;
  }

  /**
   * @brief Returns a mutable reference to the main tree.
   */
  auto main_tree() -> tree<Index, RealType, Dims> & { return _main_tree; }

  /**
   * @brief Returns a const reference to the delta tree.
   */
  auto delta_tree() const -> const tree<Index, RealType, Dims> & {
    return _delta_tree;
  }

  /**
   * @brief Returns a mutable reference to the delta tree.
   */
  auto delta_tree() -> tree<Index, RealType, Dims> & { return _delta_tree; }

  /**
   * @brief Clears all data from both the main and delta trees.
   */
  auto clear() {
    _main_tree.clear();
    _delta_tree.clear();
    _delta_ids.clear();
  }

private:
  template <typename Range, typename F>
  auto update_main_tree(const Range &id_map, const F &keep_if) {
    auto &ids = _main_tree.ids();
    tf::parallel_apply(_main_tree.nodes(), [&](auto &node) {
      if (!node.is_leaf())
        return;
      auto &data = node.get_data();
      auto c_ids = tf::make_range(ids.begin() + data[0], data[1]);
      // we apply the index_map
      for (auto &x : c_ids)
        x = id_map[x];
      // we move invalidated ids to the end of the subrange
      // and set new size
      node.set_data(data[0],
                    std::partition(c_ids.begin(), c_ids.end(), keep_if) -
                        c_ids.begin());
    });
  }
  template <typename F> auto update_main_tree(const F &keep_if) {
    auto &ids = _main_tree.ids();
    tf::parallel_apply(_main_tree.nodes(), [&](auto &node) {
      if (!node.is_leaf())
        return;
      auto &data = node.get_data();
      auto c_ids = tf::make_range(ids.begin() + data[0], data[1]);
      // we move invalidated ids to the end of the subrange
      // and set new size
      node.set_data(data[0],
                    std::partition(c_ids.begin(), c_ids.end(), keep_if) -
                        c_ids.begin());
    });
  }

  template <typename Range0, typename Range1, typename F, typename FC>
  auto update_delta_tree(const Range0 &objects, const Range1 &ids,
                         const F &keep_if, const tf::tree_config<FC> &config) {
    auto n_additional_objects = ids.size();
    _delta_ids.allocate(n_additional_objects + _delta_ids.size());
    // keep all old ids that are not in the
    // other region (as those will get copied from there)
    auto write_to =
        std::copy_if(_delta_tree.ids().begin(), _delta_tree.ids().end(),
                     _delta_ids.begin(), keep_if);
    write_to = std::copy(ids.begin(), ids.end(), write_to);
    // this will only bump down the end pointer
    _delta_ids.allocate(write_to - _delta_ids.begin());
    _delta_tree.build(tf::make_indirect_range(_delta_ids, objects), config);
    // ids in the tree are a permutation of
    // [0, ..., _small_ids.size()]
    // so we remap them to the real ids
    tf::parallel_copy(tf::make_indirect_range(_delta_tree.ids(), _delta_ids),
                      _delta_tree.ids());
  }

  template <typename Range, typename Range1, typename Range2, typename F,
            typename FC>
  auto update_delta_tree(const Range &objects,
                         const tf::index_map<Range1, Range2> &index_map,
                         const F &keep_if, const tf::tree_config<FC> &config) {
    auto n_additional_objects = index_map.kept_ids().size();
    _delta_ids.allocate(n_additional_objects + _delta_ids.size());
    // keep all old ids that are not in the
    // other region (as those will get copied from there)
    auto mapped_ids = tf::make_indirect_range(_delta_tree.ids(), index_map.f());
    auto write_to = std::copy_if(mapped_ids.begin(), mapped_ids.end(),
                                 _delta_ids.begin(), keep_if);
    write_to = std::copy(index_map.kept_ids().begin(),
                         index_map.kept_ids().end(), write_to);
    // this will only bump down the end pointer
    _delta_ids.allocate(write_to - _delta_ids.begin());
    _delta_tree.build(tf::make_indirect_range(_delta_ids, objects), config);
    // ids in the tree are a permutation of
    // [0, ..., _small_ids.size()]
    // so we remap them to the real ids
    tf::parallel_copy(tf::make_indirect_range(_delta_tree.ids(), _delta_ids),
                      _delta_tree.ids());
  }

  tree<Index, RealType, Dims> _main_tree;
  tree<Index, RealType, Dims> _delta_tree;
  tf::buffer<Index> _delta_ids;
};
} // namespace tf
