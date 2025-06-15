/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./implementation/search.hpp"

namespace tf {
template <std::size_t N, typename Policy, typename F0, typename F1>
auto search(const tf::form<N, Policy> &form, const F0 &check_aabb,
            const F1 &primitive_apply) -> bool {
  return tf::implementation::search(form, check_aabb, primitive_apply);
}

template <std::size_t N, typename Policy0, typename Policy1, typename F0,
          typename F1>
auto search(const tf::form<N, Policy0> &form0,
            const tf::form<N, Policy1> &form1, const F0 &check_aabbs,
            const F1 &primitive_apply) -> bool {
  return implementation::dual_form_search_dispatch<typename Policy0::index_t>(
      form0, form1, check_aabbs, primitive_apply, 6);
}

/// @brief Perform a spatial query against a single tree structure.
///
/// Iterates through the tree and applies a user-provided callback to all
/// primitive IDs whose AABBs intersect the query condition.
///
/// @param tree The spatial tree to search.
/// @param check_aabb A predicate that determines whether a node's AABB should
/// be traversed.
///                   Signature: `bool(const tf::aabb<RealT, N>& aabb)`
/// @param primitive_apply A function applied to each matching primitive ID.
/// Return `true` to abort early.
///                        Signature: `(Index id) -> bool`
///
/// @return bool
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto search(const tf::tree<Index, RealT, N> &tree, const F0 &check_aabb,
            const F1 &primitive_apply) -> bool {
  return tf::implementation::search(tree, check_aabb, primitive_apply);
}

/// @brief Perform a spatial query against a single tree structure.
///
/// Iterates through the tree and applies a user-provided callback to all
/// primitive IDs whose AABBs intersect the query condition.
///
/// @param tree The spatial tree to search.
/// @param check_aabb A predicate that determines whether a node's AABB should
/// be traversed.
///                   Signature: `bool(const tf::aabb<RealT, N>& aabb)`
/// @param primitive_apply A function applied to each matching primitive ID.
/// Return `true` to abort early.
///                        Signature: `(Index id) -> bool`
///
/// @return bool
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto search(const tf::mod_tree<Index, RealT, N> &tree, const F0 &check_aabb,
            const F1 &primitive_apply) -> bool {
  if (!search(tree.main_tree(), check_aabb, primitive_apply))
    return search(tree.delta_tree(), check_aabb, primitive_apply);
  else
    return true;
}

/// @brief Perform a parallel pairwise search between two spatial trees.
///
///
/// @param tree0 The first spatial tree.
/// @param tree1 The second spatial tree.
/// @param check_aabbs Predicate that decides whether to recurse into a pair of
/// nodes.
///                    Signature: `(const tf::aabb<RealT, N>& aabb0, const
///                    tf::aabb<RealT, N>& aabb1) -> bool`
/// @param primitive_apply Function called for each pair of primitive IDs in
/// intersecting leaves.
///                        Signature: `(Index id0, Index id1) -> bool`
///                        **Must be thread-safe** if it accesses shared memory.
///
/// @return bool
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto search(const tf::tree<Index, RealT, N> &tree0,
            const tf::tree<Index, RealT, N> &tree1, const F0 &check_aabbs,
            const F1 &primitive_apply, int paralelism_depth = 6) -> bool {
  return implementation::dual_search_dispatch<Index>(
      tree0, tree1, check_aabbs, primitive_apply, paralelism_depth);
}

/// @brief Perform a parallel pairwise search between two spatial trees.
///
///
/// @param tree0 The first spatial tree.
/// @param tree1 The second spatial tree.
/// @param check_aabbs Predicate that decides whether to recurse into a pair of
/// nodes.
///                    Signature: `(const tf::aabb<RealT, N>& aabb0, const
///                    tf::aabb<RealT, N>& aabb1) -> bool`
/// @param primitive_apply Function called for each pair of primitive IDs in
/// intersecting leaves.
///                        Signature: `(Index id0, Index id1) -> bool`
///                        **Must be thread-safe** if it accesses shared memory.
///
/// @return bool
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto search(const tf::mod_tree<Index, RealT, N> &tree0,
            const tf::tree<Index, RealT, N> &tree1, const F0 &check_aabbs,
            const F1 &primitive_apply, int paralelism_depth = 6) -> bool {
  return implementation::dual_search_dispatch<Index>(
      tree0, tree1, check_aabbs, primitive_apply, paralelism_depth);
}

/// @brief Perform a parallel pairwise search between two spatial trees.
///
///
/// @param tree0 The first spatial tree.
/// @param tree1 The second spatial tree.
/// @param check_aabbs Predicate that decides whether to recurse into a pair of
/// nodes.
///                    Signature: `(const tf::aabb<RealT, N>& aabb0, const
///                    tf::aabb<RealT, N>& aabb1) -> bool`
/// @param primitive_apply Function called for each pair of primitive IDs in
/// intersecting leaves.
///                        Signature: `(Index id0, Index id1) -> bool`
///                        **Must be thread-safe** if it accesses shared memory.
///
/// @return bool
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto search(const tf::tree<Index, RealT, N> &tree0,
            const tf::mod_tree<Index, RealT, N> &tree1, const F0 &check_aabbs,
            const F1 &primitive_apply, int paralelism_depth = 6) -> bool {
  return implementation::dual_search_dispatch<Index>(
      tree0, tree1, check_aabbs, primitive_apply, paralelism_depth);
}

/// @brief Perform a parallel pairwise search between two spatial trees.
///
///
/// @param tree0 The first spatial tree.
/// @param tree1 The second spatial tree.
/// @param check_aabbs Predicate that decides whether to recurse into a pair of
/// nodes.
///                    Signature: `(const tf::aabb<RealT, N>& aabb0, const
///                    tf::aabb<RealT, N>& aabb1) -> bool`
/// @param primitive_apply Function called for each pair of primitive IDs in
/// intersecting leaves.
///                        Signature: `(Index id0, Index id1) -> bool`
///                        **Must be thread-safe** if it accesses shared memory.
///
/// @return bool
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto search(const tf::mod_tree<Index, RealT, N> &tree0,
            const tf::mod_tree<Index, RealT, N> &tree1, const F0 &check_aabbs,
            const F1 &primitive_apply, int paralelism_depth = 6) -> bool {
  return implementation::dual_search_dispatch<Index>(
      tree0, tree1, check_aabbs, primitive_apply, paralelism_depth);
}

} // namespace tf
