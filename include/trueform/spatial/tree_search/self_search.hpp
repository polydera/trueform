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
#include "../../core/frame_of.hpp"
#include "../../core/policy/id.hpp"
#include "../../core/transformed.hpp"
#include "../../core/form.hpp"
#include "../tree/self_search.hpp"
#include <atomic>

namespace tf::spatial {

// ============================================================================
// tree_like self search
// ============================================================================

template <typename TreePolicy, typename F0, typename F1, typename F2>
auto search_self(const tf::tree_like<TreePolicy> &tree, const F0 &check_bvs,
                 const F1 &primitive_apply, const F2 &abort,
                 int parallelism_depth = 6) -> bool {
  using Index = typename TreePolicy::index_type;
  return tf::spatial::impl::self_search(
      tree, check_bvs,
      [primitive_apply, &tree, &check_bvs](const auto &ids0, const auto &ids1,
                                            bool is_self) {
        for (Index i0 = 0; i0 < Index(ids0.size()); ++i0) {
          auto id0 = ids0[i0];
          for (Index i1 = (i0 + 1) * is_self; i1 < Index(ids1.size()); ++i1) {
            auto id1 = ids1[i1];
            if (check_bvs(tree.primitive_aabbs()[id0],
                          tree.primitive_aabbs()[id1]) &&
                primitive_apply(id0, id1))
              return true;
          }
        }
        return false;
      },
      abort, parallelism_depth);
}

// ============================================================================
// form self search
// ============================================================================

template <std::size_t Dims, typename Policy, typename F0, typename F1,
          typename F2>
auto search_self(const tf::form<Dims, Policy> &form, const F0 &check_bvs,
                 const F1 &primitive_apply, const F2 &abort,
                 int parallelism_depth = 6) -> bool {
  using Index = typename Policy::index_type;
  auto bv_f = [&](const auto &bv0, const auto &bv1) -> bool {
    return check_bvs(tf::transformed(bv0, tf::transformation_of(form)),
                     tf::transformed(bv1, tf::transformation_of(form)));
  };
  return tf::spatial::impl::self_search(
      form.tree(), bv_f,
      [primitive_apply, &form, &bv_f](const auto &ids0, const auto &ids1,
                                       bool is_self) {
        for (Index i0 = 0; i0 < Index(ids0.size()); ++i0) {
          auto id0 = ids0[i0];
          auto obj0 = tf::tag_id(
              id0, tf::transformed(form[id0], tf::transformation_of(form)));
          for (Index i1 = (i0 + 1) * is_self; i1 < Index(ids1.size()); ++i1) {
            auto id1 = ids1[i1];
            auto obj1 = tf::tag_id(
                id1, tf::transformed(form[id1], tf::transformation_of(form)));
            if (bv_f(form.tree().primitive_aabbs()[id0],
                     form.tree().primitive_aabbs()[id1]) &&
                primitive_apply(obj0, obj1))
              return true;
          }
        }
        return false;
      },
      abort, parallelism_depth);
}

// ============================================================================
// Dispatch helpers (wrap abort handling)
// ============================================================================

template <typename TreePolicy, typename F0, typename F1>
auto search_self_dispatch(const tf::tree_like<TreePolicy> &tree,
                          const F0 &check_bvs, const F1 &primitive_apply,
                          int parallelism_depth = 6) -> bool {
  using Index = typename TreePolicy::index_type;

  if constexpr (!std::is_same_v<decltype(primitive_apply(Index(0), Index(0))),
                                void>) {
    std::atomic_bool flag{false};
    auto abort_f = [&flag] { return flag.load(); };
    auto apply_f = [&flag, primitive_apply](Index id0, Index id1) -> bool {
      if (primitive_apply(id0, id1)) {
        flag.store(true);
        return true;
      }
      return false;
    };
    return spatial::search_self(tree, check_bvs, apply_f, abort_f,
                                parallelism_depth);
  } else {
    auto apply_f = [primitive_apply](Index id0, Index id1) -> bool {
      primitive_apply(id0, id1);
      return false;
    };
    auto abort_f = [] { return false; };
    return spatial::search_self(tree, check_bvs, apply_f, abort_f,
                                parallelism_depth);
  }
}

template <std::size_t Dims, typename Policy, typename F0, typename F1>
auto search_self_form_dispatch(const tf::form<Dims, Policy> &form,
                               const F0 &check_bvs, const F1 &primitive_apply,
                               int parallelism_depth = 6) -> bool {
  using Index = typename Policy::index_type;

  if constexpr (!std::is_same_v<
                    decltype(primitive_apply(
                        tf::tag_id(Index(0),
                                   tf::transformed(form[Index(0)],
                                                   tf::transformation_of(form))),
                        tf::tag_id(Index(0),
                                   tf::transformed(form[Index(0)],
                                                   tf::transformation_of(form))))),
                    void>) {
    std::atomic_bool flag{false};
    auto abort_f = [&flag] { return flag.load(); };
    auto apply_f = [&flag, primitive_apply](auto &&obj0, auto &&obj1) -> bool {
      if (primitive_apply(obj0, obj1)) {
        flag.store(true);
        return true;
      }
      return false;
    };
    return spatial::search_self(form, check_bvs, apply_f, abort_f,
                                parallelism_depth);
  } else {
    auto apply_f = [primitive_apply](auto &&obj0, auto &&obj1) -> bool {
      primitive_apply(obj0, obj1);
      return false;
    };
    auto abort_f = [] { return false; };
    return spatial::search_self(form, check_bvs, apply_f, abort_f,
                                parallelism_depth);
  }
}

} // namespace tf::spatial
