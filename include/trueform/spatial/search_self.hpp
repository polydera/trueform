/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./tree_search/self_search.hpp"

namespace tf {

/// @brief Perform a self-intersection search on a tree_like structure.
///
/// Finds all pairs of primitives within the same tree whose bounding volumes
/// intersect.
///
/// @param tree The tree_like to search.
/// @param check_bvs Predicate that decides whether to recurse into a pair of
/// nodes.
/// @param primitive_apply Function called for each pair of primitive IDs.
/// **Must be thread-safe** if it accesses shared memory.
/// @param parallelism_depth Depth at which to spawn parallel tasks.
///
/// @return bool
template <typename TreePolicy, typename F0, typename F1>
auto search_self(const tf::tree_like<TreePolicy> &tree, const F0 &check_bvs,
                 const F1 &primitive_apply, int parallelism_depth = 6) -> bool {
  return spatial::search_self_dispatch(tree, check_bvs, primitive_apply,
                                       parallelism_depth);
}

/// @brief Perform a self-intersection search on a form.
///
/// Finds all pairs of primitives within the same form whose bounding volumes
/// intersect.
///
/// @param form The form to search.
/// @param check_bvs Predicate that decides whether to recurse into a pair of
/// nodes.
/// @param primitive_apply Function called for each pair of primitives.
/// **Must be thread-safe** if it accesses shared memory.
/// @param parallelism_depth Depth at which to spawn parallel tasks.
///
/// @return bool
template <std::size_t Dims, typename Policy, typename F0, typename F1>
auto search_self(const tf::form<Dims, Policy> &form, const F0 &check_bvs,
                 const F1 &primitive_apply, int parallelism_depth = 6) -> bool {
  return spatial::search_self_form_dispatch(form, check_bvs, primitive_apply,
                                            parallelism_depth);
}

} // namespace tf
