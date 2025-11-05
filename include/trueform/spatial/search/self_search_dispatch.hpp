/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/policy/id.hpp"
#include "../../core/transformed.hpp"
#include "../form.hpp"
#include "../tree.hpp"
#include "./tree_self_search.hpp"
#include <atomic>
namespace tf::spatial {
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1, typename F2>
auto search_self(const tf::tree<Index, RealT, N> &tree, const F0 &check_aabbs,
                 const F1 &primitive_apply, const F2 &abort,
                 int paralelism_depth = 6) -> bool {
  return tf::spatial::tree_self_search(
      tree.nodes(), tree.ids(), check_aabbs,
      [primitive_apply, &tree, &check_aabbs](const auto &ids0, const auto &ids1,
                                             bool is_self) {
        for (Index i0 = 0; i0 < Index(ids0.size()); ++i0) {
          auto id0 = ids0[i0];
          for (Index i1 = (i0 + 1) * is_self; i1 < Index(ids1.size()); ++i1) {
            auto id1 = ids1[i1];
            if (check_aabbs(tree.primitive_aabbs()[id0],
                            tree.primitive_aabbs()[id1]) &&
                primitive_apply(id0, id1))
              return true;
          }
        }
        return false;
      },
      abort, paralelism_depth);
}

template <std::size_t N, typename Policy, typename F0, typename F1, typename F2>
auto search_self(const tf::form<N, Policy> &form, const F0 &check_aabbs,
                 const F1 &primitive_apply, const F2 &abort,
                 int paralelism_depth = 6) -> bool {
  using Index = typename Policy::index_t;
  auto aabb_f = [&](const auto &aabb0, const auto &aabb1) -> bool {
    return check_aabbs(tf::transformed(aabb0, form.transformation()),
                       tf::transformed(aabb1, form.transformation()));
  };
  return tf::spatial::tree_self_search(
      form.tree().nodes(), form.tree().ids(), aabb_f,
      [primitive_apply, &form, &aabb_f](const auto &ids0, const auto &ids1,
                                        bool is_self) {
        for (Index i0 = 0; i0 < Index(ids0.size()); ++i0) {
          auto id0 = ids0[i0];
          auto obj0 = tf::tag_id(
              id0, tf::transformed(form[id0], form.transformation()));
          for (Index i1 = (i0 + 1) * is_self; i1 < Index(ids1.size()); ++i1) {
            auto id1 = ids1[i1];
            auto obj1 = tf::tag_id(
                id1, tf::transformed(form[id1], form.transformation()));
            if (aabb_f(form.tree().primitive_aabbs()[id0],
                       form.tree().primitive_aabbs()[id1]) &&
                primitive_apply(obj0, obj1))
              return true;
          }
        }
        return false;
      },
      abort, paralelism_depth);
}

template <typename Index, typename Tree, typename F0, typename F1>
auto search_self_dispatch(const Tree &tree, const F0 &check_aabbs,
                          const F1 &primitive_apply, int paralelism_depth = 6)
    -> bool {
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
    return spatial::search_self(tree, check_aabbs, apply_f, abort_f,
                                paralelism_depth);
  } else {
    auto apply_f = [primitive_apply](Index id0, Index id1) -> bool {
      primitive_apply(id0, id1);
      return false;
    };
    auto abort_f = [] { return false; };
    return spatial::search_self(tree, check_aabbs, apply_f, abort_f,
                                paralelism_depth);
  }
}

template <typename Index, typename Form, typename F0, typename F1>
auto search_self_form_dispatch(const Form &form, const F0 &check_aabbs,
                               const F1 &primitive_apply,
                               int paralelism_depth = 6) -> bool {

  if constexpr (!std::is_same_v<
                    decltype(primitive_apply(
                        tf::tag_id(Index(0),
                                   tf::transformed(form[Index(0)],
                                                   form.transformation())),
                        tf::tag_id(Index(0),
                                   tf::transformed(form[Index(0)],
                                                   form.transformation())))),
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
    return spatial::search_self(form, check_aabbs, apply_f, abort_f,
                                paralelism_depth);
  } else {
    auto apply_f = [primitive_apply](auto &&obj0, auto &&obj1) -> bool {
      primitive_apply(obj0, obj1);
      return false;
    };
    auto abort_f = [] { return false; };
    return spatial::search_self(form, check_aabbs, apply_f, abort_f,
                                paralelism_depth);
  }
}
} // namespace tf::spatial
