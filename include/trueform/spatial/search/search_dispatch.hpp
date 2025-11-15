/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/policy/id.hpp"
#include "../../core/transformed.hpp"
#include "../form.hpp"
#include "../tree.hpp"
#include "./tree_dual_search.hpp"
#include "./tree_search.hpp"
#include <atomic>

namespace tf::spatial {
template <std::size_t N, typename Policy, typename F0, typename F1>
auto search(const tf::form<N, Policy> &form, const F0 &check_aabb,
            const F1 &primitive_apply) -> bool {
  return tf::spatial::tree_search(
      form.tree().nodes(), form.tree().ids(),
      [&check_aabb, &form](const auto &aabb) {
        return check_aabb(tf::transformed(aabb, form.frame()));
      },
      [primitive_apply, &form, &check_aabb](const auto &r) {
        for (const auto &id : r)
          if (check_aabb(tf::transformed(form.tree().primitive_aabbs()[id],
                                         form.frame()))) {
            if constexpr (std::is_same_v<decltype(primitive_apply(tf::tag_id(
                                             id, tf::transformed(
                                                     form[id], form.frame())))),
                                         void>) {
              primitive_apply(
                  tf::tag_id(id, tf::transformed(form[id], form.frame())));
            } else {
              if (primitive_apply(
                      tf::tag_id(id, tf::transformed(form[id], form.frame()))))
                return true;
            }
          }
        return false;
      });
}

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto search(const tf::tree<Index, RealT, N> &tree, const F0 &check_aabb,
            const F1 &primitive_apply) -> bool {
  return tf::spatial::tree_search(
      tree.nodes(), tree.ids(), check_aabb,
      [primitive_apply, &tree, &check_aabb](const auto &r) {
        for (const auto &id : r)
          if (check_aabb(tree.primitive_aabbs()[id])) {
            if constexpr (std::is_same_v<decltype(primitive_apply(id)), void>) {
              primitive_apply(id);
            } else {
              if (primitive_apply(id))
                return true;
            }
          }
        return false;
      });
}

template <typename Index0, typename Index1, typename RealT, std::size_t N,
          typename F0, typename F1, typename F2>
auto search(const tf::tree<Index0, RealT, N> &tree0,
            const tf::tree<Index1, RealT, N> &tree1, const F0 &check_aabbs,
            const F1 &primitive_apply, const F2 &abort,
            int paralelism_depth = 6) -> bool {
  return tf::spatial::tree_dual_search(
      tree0.nodes(), tree0.ids(), tree1.nodes(), tree1.ids(), check_aabbs,
      [primitive_apply, &tree0, &tree1, &check_aabbs](const auto &r0,
                                                      const auto &r1) {
        for (const auto &id0 : r0)
          for (const auto &id1 : r1)
            if (check_aabbs(tree0.primitive_aabbs()[id0],
                            tree1.primitive_aabbs()[id1]) &&
                primitive_apply(id0, id1))
              return true;
        return false;
      },
      abort, paralelism_depth);
}

template <std::size_t N, typename Policy0, typename Policy1, typename F0,
          typename F1, typename F2>
auto search(const tf::form<N, Policy0> &form0,
            const tf::form<N, Policy1> &form1, const F0 &check_aabbs,
            const F1 &primitive_apply, const F2 &abort,
            int paralelism_depth = 6) -> bool {
  auto aabb_f = [&](const auto &aabb0, const auto &aabb1) -> bool {
    return check_aabbs(tf::transformed(aabb0, form0.frame()),
                       tf::transformed(aabb1, form1.frame()));
  };
  return tf::spatial::tree_dual_search(
      form0.tree().nodes(), form0.tree().ids(), form1.tree().nodes(),
      form1.tree().ids(), aabb_f,
      [primitive_apply, &form0, &form1, &aabb_f](const auto &r0,
                                                 const auto &r1) {
        for (const auto &id0 : r0) {
          auto obj0 =
              tf::tag_id(id0, tf::transformed(form0[id0], form0.frame()));
          for (const auto &id1 : r1)
            if (aabb_f(form0.tree().primitive_aabbs()[id0],
                       form1.tree().primitive_aabbs()[id1]) &&
                primitive_apply(
                    obj0, tf::tag_id(
                              id1, tf::transformed(form1[id1], form1.frame()))))
              return true;
        }
        return false;
      },
      abort, paralelism_depth);
}

/*template <typename Index, typename RealT, std::size_t N, typename F0,*/
/*          typename F1, typename F2>*/
/*auto search(const tf::mod_tree<Index, RealT, N> &tree0,*/
/*            const tf::tree<Index, RealT, N> &tree1, const F0 &check_aabbs,*/
/*            const F1 &primitive_apply, const F2 &abort,*/
/*            int paralelism_depth = 6) -> bool {*/
/*  if (!search(tree0.main_tree(), tree1, check_aabbs, primitive_apply, abort,*/
/*              paralelism_depth))*/
/*    return search(tree0.delta_tree(), tree1, check_aabbs, primitive_apply,*/
/*                  abort, paralelism_depth);*/
/*  else*/
/*    return true;*/
/*}*/
/**/
/*template <typename Index, typename RealT, std::size_t N, typename F0,*/
/*          typename F1, typename F2>*/
/*auto search(const tf::tree<Index, RealT, N> &tree0,*/
/*            const tf::mod_tree<Index, RealT, N> &tree1, const F0
 * &check_aabbs,*/
/*            const F1 &primitive_apply, const F2 &abort,*/
/*            int paralelism_depth = 6) -> bool {*/
/*  if (!search(tree0, tree1.main_tree(), check_aabbs, primitive_apply, abort,*/
/*              paralelism_depth))*/
/*    return search(tree0, tree1.delta_tree(), check_aabbs, primitive_apply,*/
/*                  abort, paralelism_depth);*/
/*  else*/
/*    return true;*/
/*}*/

template <typename Tree0, typename Tree1, typename F0, typename F1>
auto dual_search_dispatch(const Tree0 &tree0, const Tree1 &tree1,
                          const F0 &check_aabbs, const F1 &primitive_apply,
                          int paralelism_depth = 6) -> bool {

  using Index0 = typename Tree0::index_t;
  using Index1 = typename Tree1::index_t;
  if constexpr (!std::is_same_v<decltype(primitive_apply(Index0(0), Index1(0))),
                                void>) {
    std::atomic_bool flag{false};
    auto abort_f = [&flag] { return flag.load(); };
    auto apply_f = [&flag, primitive_apply](Index0 id0, Index1 id1) -> bool {
      if (primitive_apply(id0, id1)) {
        flag.store(true);
        return true;
      }
      return false;
    };
    return spatial::search(tree0, tree1, check_aabbs, apply_f, abort_f,
                           paralelism_depth);
  } else {
    auto apply_f = [primitive_apply](Index0 id0, Index1 id1) -> bool {
      primitive_apply(id0, id1);
      return false;
    };
    auto abort_f = [] { return false; };
    return spatial::search(tree0, tree1, check_aabbs, apply_f, abort_f,
                           paralelism_depth);
  }
}
// this is "duplicated", because in the form case we additionally
// optimize in the leaves, where we only transform once per
// outer loop, instead of always in the applier
template <typename Form0, typename Form1, typename F0, typename F1>
auto dual_form_search_dispatch(const Form0 &form0, const Form1 &form1,
                               const F0 &check_aabbs, const F1 &primitive_apply,
                               int paralelism_depth = 6) -> bool {

  using Index0 = typename Form0::index_t;
  using Index1 = typename Form1::index_t;
  if constexpr (!std::is_same_v<
                    decltype(primitive_apply(
                        tf::tag_id(Index0(0), tf::transformed(form0[Index0(0)],
                                                              form0.frame())),
                        tf::tag_id(Index1(0), tf::transformed(form1[Index1(0)],
                                                              form1.frame())))),
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
    return spatial::search(form0, form1, check_aabbs, apply_f, abort_f,
                           paralelism_depth);
  } else {
    auto apply_f = [primitive_apply](auto &&obj0, auto &&obj1) -> bool {
      primitive_apply(obj0, obj1);
      return false;
    };
    auto abort_f = [] { return false; };
    return spatial::search(form0, form1, check_aabbs, apply_f, abort_f,
                           paralelism_depth);
  }
}

} // namespace tf::spatial
