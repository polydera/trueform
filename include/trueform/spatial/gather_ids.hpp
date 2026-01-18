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
#include "./search.hpp"

namespace tf {

/// @ingroup spatial_queries
/// @brief Gather pairs of primitive IDs from two forms that satisfy predicates.
///
/// Performs a dual-tree search and collects pairs of IDs where both the
/// bounding volume predicate and primitive predicate are satisfied.
///
/// @note Pair search runs in parallel. Use @ref tf::local_value or
/// @ref tf::local_vector for thread-safe aggregation in custom predicates.
///
/// @param form0 The first form.
/// @param form1 The second form.
/// @param aabbs_predicate Predicate for bounding volume pairs.
/// @param primitives_predicate Predicate for primitive pairs.
/// @param out Output iterator to write pairs of IDs.
/// @return Iterator past the last written element.
template <std::size_t Dims, typename Policy0, typename Policy1, typename F0,
          typename F1, typename Iterator>
auto gather_ids(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1, const F0 &aabbs_predicate,
                const F1 &primitives_predicate, Iterator out) -> Iterator {
  static_assert(tf::has_tree_policy<Policy0>,
                "First form must have a tree policy attached. Use: form | tf::tag(tree)");
  static_assert(tf::has_tree_policy<Policy1>,
                "Second form must have a tree policy attached. Use: form | tf::tag(tree)");
  auto get_index_t = [](auto form) {
    if constexpr (tf::has_id_policy<decltype(form[0])>)
      return form[0].id();
    else
      return typename decltype(form)::index_type(0);
  };
  using index_t0 = std::decay_t<decltype(get_index_t(form0))>;
  using index_t1 = std::decay_t<decltype(get_index_t(form1))>;
  struct holder_t {
    index_t0 id0;
    index_t1 id1;
  };
  tf::local_vector<holder_t> l_ids;
  tf::search(form0, form1, aabbs_predicate,
             [&](const auto &obj0, const auto &obj1) {
               if (primitives_predicate(obj0, obj1))
                 l_ids.push_back(holder_t{obj0.id(), obj1.id()});
             });
  for (const auto &v : l_ids.vectors())
    for (const auto &e : v)
      *out++ = std::make_pair(e.id0, e.id1);
  return out;
}

/// @ingroup spatial_queries
/// @brief Gather pairs of primitive IDs using a single predicate for both tests.
template <std::size_t Dims, typename Policy0, typename Policy1, typename F,
          typename Iterator>
auto gather_ids(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1, const F &predicate,
                Iterator out) -> Iterator {
  return gather_ids(form0, form1, predicate, predicate, out);
}

/// @ingroup spatial_queries
/// @brief Gather primitive IDs from a form that satisfy predicates.
///
/// @param form The form to query.
/// @param aabb_predicate Predicate for bounding volumes.
/// @param primitive_predicate Predicate for primitives.
/// @param out Output iterator to write IDs.
/// @return Iterator past the last written element.
template <std::size_t Dims, typename Policy, typename F0, typename F1,
          typename Iterator>
auto gather_ids(const tf::form<Dims, Policy> &form, const F0 &aabb_predicate,
                const F1 &primitive_predicate, Iterator out) -> Iterator {
  static_assert(tf::has_tree_policy<Policy>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  tf::search(form, aabb_predicate, [&](const auto &obj) {
    if (primitive_predicate(obj))
      *out++ = obj.id();
  });
  return out;
}

/// @ingroup spatial_queries
/// @brief Gather primitive IDs using a single predicate for both tests.
template <std::size_t Dims, typename Policy, typename F, typename Iterator>
auto gather_ids(const tf::form<Dims, Policy> &form, const F &predicate,
                Iterator out) -> Iterator {
  return gather_ids(form, predicate, predicate, out);
}
} // namespace tf
