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
#include "./policy/tree.hpp"
#include "./tree_search/search.hpp"

namespace tf {

/// @ingroup spatial_queries
/// @brief Perform a spatial query against a form.
///
/// Iterates through the form's tree and applies a user-provided callback to all
/// primitive IDs whose bounding volumes intersect the query condition.
///
/// @param form The form to search.
/// @param check_bv A predicate that determines whether a node's BV should
/// be traversed.
/// @param primitive_apply A function applied to each matching primitive.
/// Return `true` to abort early.
///
/// @return bool
template <std::size_t Dims, typename Policy, typename F0, typename F1>
auto search(const tf::form<Dims, Policy> &form, const F0 &check_bv,
            const F1 &primitive_apply) -> bool {
  static_assert(tf::has_tree_policy<Policy>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  return tf::spatial::search(form, check_bv, primitive_apply);
}

/// @ingroup spatial_queries
/// @brief Perform a parallel pairwise search between two forms.
///
/// @param form0 The first form.
/// @param form1 The second form.
/// @param check_bvs Predicate that decides whether to recurse into a pair of
/// nodes.
/// @param primitive_apply Function called for each pair of primitives in
/// intersecting leaves. **Must be thread-safe** if it accesses shared memory.
///
/// @return bool
template <std::size_t Dims, typename Policy0, typename Policy1, typename F0,
          typename F1>
auto search(const tf::form<Dims, Policy0> &form0,
            const tf::form<Dims, Policy1> &form1, const F0 &check_bvs,
            const F1 &primitive_apply) -> bool {
  static_assert(tf::has_tree_policy<Policy0>,
                "First form must have a tree policy attached. Use: form | tf::tag(tree)");
  static_assert(tf::has_tree_policy<Policy1>,
                "Second form must have a tree policy attached. Use: form | tf::tag(tree)");
  return spatial::dual_form_search_dispatch(form0, form1, check_bvs,
                                            primitive_apply, 6);
}

/// @ingroup spatial_queries
/// @brief Perform a spatial query against a tree_like structure.
///
/// Iterates through the tree and applies a user-provided callback to all
/// primitive IDs whose bounding volumes intersect the query condition.
///
/// @param tree The tree_like to search.
/// @param check_bv A predicate that determines whether a node's BV should
/// be traversed.
/// @param primitive_apply A function applied to each matching primitive ID.
/// Return `true` to abort early.
///
/// @return bool
template <typename TreePolicy, typename F0, typename F1>
auto search(const tf::tree_like<TreePolicy> &tree, const F0 &check_bv,
            const F1 &primitive_apply) -> bool {
  return tf::spatial::search(tree, check_bv, primitive_apply);
}

/// @ingroup spatial_queries
/// @brief Perform a parallel pairwise search between two tree_like structures.
///
/// @param tree0 The first tree.
/// @param tree1 The second tree.
/// @param check_bvs Predicate that decides whether to recurse into a pair of
/// nodes.
/// @param primitive_apply Function called for each pair of primitive IDs in
/// intersecting leaves. **Must be thread-safe** if it accesses shared memory.
///
/// @return bool
template <typename TreePolicy0, typename TreePolicy1, typename F0, typename F1>
auto search(const tf::tree_like<TreePolicy0> &tree0,
            const tf::tree_like<TreePolicy1> &tree1, const F0 &check_bvs,
            const F1 &primitive_apply, int parallelism_depth = 6) -> bool {
  return spatial::dual_search_dispatch(tree0, tree1, check_bvs, primitive_apply,
                                       parallelism_depth);
}

} // namespace tf
