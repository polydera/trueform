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

#include "../core/local_vector.hpp"
#include "./policy/tree.hpp"
#include "./search_self.hpp"

namespace tf {

/// @ingroup spatial_queries
/// @brief Gather pairs of primitive IDs within a form that satisfy predicates.
///
/// Performs a self-intersection search and collects pairs of IDs where both
/// the bounding volume predicate and primitive predicate are satisfied.
///
/// @note Self-search runs in parallel. Use @ref tf::local_value or
/// @ref tf::local_vector for thread-safe aggregation in custom predicates.
///
/// @param form The form to query.
/// @param aabbs_predicate Predicate for bounding volume pairs.
/// @param primitives_predicate Predicate for primitive pairs.
/// @param out Output iterator to write pairs of IDs.
/// @return Iterator past the last written element.
template <std::size_t Dims, typename Policy, typename F0, typename F1,
          typename Iterator>
auto gather_self_ids(const tf::form<Dims, Policy> &form,
                     const F0 &aabbs_predicate, const F1 &primitives_predicate,
                     Iterator out) -> Iterator {
  static_assert(tf::has_tree_policy<Policy>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto get_index_t = [](auto form) {
    if constexpr (tf::has_id_policy<decltype(form[0])>)
      return form[0].id();
    else
      return typename decltype(form)::index_type(0);
  };
  using index_t = std::decay_t<decltype(get_index_t(form))>;
  struct holder_t {
    index_t id0;
    index_t id1;

    operator std::pair<index_t, index_t>() const { return {id0, id1}; }
    operator std::array<index_t, 2>() const { return {id0, id1}; }
  };
  tf::local_vector<holder_t> l_ids;
  tf::search_self(form, aabbs_predicate,
                  [&](const auto &obj0, const auto &obj1) {
                    if (primitives_predicate(obj0, obj1))
                      l_ids.push_back(holder_t{obj0.id(), obj1.id()});
                  });
  return l_ids.to_iterator(out);
}

/// @ingroup spatial_queries
/// @brief Gather self-intersecting pairs using a single predicate for both tests.
template <std::size_t Dims, typename Policy, typename F, typename Iterator>
auto gather_self_ids(const tf::form<Dims, Policy> &form, const F &predicate,
                     Iterator out) -> Iterator {
  return gather_self_ids(form, predicate, predicate, out);
}
} // namespace tf
