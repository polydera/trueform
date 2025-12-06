/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/make_buffer_for_transformed.hpp"
#include "../../core/make_local_buffer_for_transformed.hpp"
#include "../../core/policy/buffer.hpp"
#include "../../core/policy/id.hpp"
#include "../../core/transformed.hpp"
#include "../form.hpp"
#include "../tree/dual_search.hpp"
#include "../tree/search.hpp"
#include <atomic>

namespace tf::spatial {

// ============================================================================
// Single tree_like search
// ============================================================================

template <typename TreePolicy, typename F0, typename F1>
auto search(const tf::tree_like<TreePolicy> &tree, const F0 &check_bv,
            const F1 &primitive_apply) -> bool {
  return tf::spatial::impl::search(
      tree, check_bv, [primitive_apply, &tree, &check_bv](const auto &r) {
        for (const auto &id : r)
          if (check_bv(tree.primitive_aabbs()[id])) {
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

// ============================================================================
// Single form search
// ============================================================================

template <std::size_t Dims, typename Policy, typename F0, typename F1>
auto search(const tf::form<Dims, Policy> &form, const F0 &check_bv,
            const F1 &primitive_apply) -> bool {
  auto buff = tf::core::make_buffer_for_transformed(form[0], form.frame());
  return tf::spatial::impl::search(
      form.tree(),
      [&check_bv, &form](const auto &bv) {
        return check_bv(tf::transformed(bv, form.frame()));
      },
      [primitive_apply, &form, &check_bv, &buff](const auto &r) {
        for (const auto &id : r)
          if (check_bv(tf::transformed(form.tree().primitive_aabbs()[id],
                                       form.frame()))) {
            if constexpr (std::is_same_v<decltype(primitive_apply(tf::tag_id(
                                             id, tf::transformed(
                                                     form[id] | tf::tag(buff),
                                                     form.frame())))),
                                         void>) {
              primitive_apply(tf::tag_id(
                  id, tf::transformed(form[id] | tf::tag(buff), form.frame())));
            } else {
              if (primitive_apply(
                      tf::tag_id(id, tf::transformed(form[id] | tf::tag(buff),
                                                     form.frame()))))
                return true;
            }
          }
        return false;
      });
}

// ============================================================================
// Dual tree_like search
// ============================================================================

template <typename TreePolicy0, typename TreePolicy1, typename F0, typename F1,
          typename F2>
auto search(const tf::tree_like<TreePolicy0> &tree0,
            const tf::tree_like<TreePolicy1> &tree1, const F0 &check_bvs,
            const F1 &primitive_apply, const F2 &abort,
            int parallelism_depth = 6) -> bool {
  return tf::spatial::impl::dual_search(
      tree0, tree1, check_bvs,
      [primitive_apply, &tree0, &tree1, &check_bvs](const auto &r0,
                                                    const auto &r1) {
        for (const auto &id0 : r0)
          for (const auto &id1 : r1)
            if (check_bvs(tree0.primitive_aabbs()[id0],
                          tree1.primitive_aabbs()[id1]) &&
                primitive_apply(id0, id1))
              return true;
        return false;
      },
      abort, parallelism_depth);
}

// ============================================================================
// Dual form search
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1, typename F0,
          typename F1, typename F2>
auto search(const tf::form<Dims, Policy0> &form0,
            const tf::form<Dims, Policy1> &form1, const F0 &check_bvs,
            const F1 &primitive_apply, const F2 &abort,
            int parallelism_depth = 6) -> bool {
  auto bv_f = [&](const auto &bv0, const auto &bv1) -> bool {
    return check_bvs(tf::transformed(bv0, form0.frame()),
                     tf::transformed(bv1, form1.frame()));
  };
  auto buff0 =
      tf::core::make_local_buffer_for_transformed(form0[0], form0.frame());
  auto buff1 =
      tf::core::make_local_buffer_for_transformed(form1[0], form1.frame());
  return tf::spatial::impl::dual_search(
      form0.tree(), form1.tree(), bv_f,
      [primitive_apply, &form0, &form1, &bv_f, &buff0, &buff1](const auto &r0,
                                                               const auto &r1) {
        for (const auto &id0 : r0) {
          auto obj0 = tf::tag_id(
              id0, tf::transformed(form0[id0] | tf::tag(buff0), form0.frame()));
          for (const auto &id1 : r1)
            if (bv_f(form0.tree().primitive_aabbs()[id0],
                     form1.tree().primitive_aabbs()[id1]) &&
                primitive_apply(
                    obj0,
                    tf::tag_id(id1, tf::transformed(form1[id1] | tf::tag(buff1),
                                                    form1.frame()))))
              return true;
        }
        return false;
      },
      abort, parallelism_depth);
}

// ============================================================================
// Dispatch helpers (wrap abort handling)
// ============================================================================

template <typename TreePolicy0, typename TreePolicy1, typename F0, typename F1>
auto dual_search_dispatch(const tf::tree_like<TreePolicy0> &tree0,
                          const tf::tree_like<TreePolicy1> &tree1,
                          const F0 &check_bvs, const F1 &primitive_apply,
                          int parallelism_depth = 6) -> bool {
  using Index0 = typename TreePolicy0::index_type;
  using Index1 = typename TreePolicy1::index_type;

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
    return spatial::search(tree0, tree1, check_bvs, apply_f, abort_f,
                           parallelism_depth);
  } else {
    auto apply_f = [primitive_apply](Index0 id0, Index1 id1) -> bool {
      primitive_apply(id0, id1);
      return false;
    };
    auto abort_f = [] { return false; };
    return spatial::search(tree0, tree1, check_bvs, apply_f, abort_f,
                           parallelism_depth);
  }
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename F0,
          typename F1>
auto dual_form_search_dispatch(const tf::form<Dims, Policy0> &form0,
                               const tf::form<Dims, Policy1> &form1,
                               const F0 &check_bvs, const F1 &primitive_apply,
                               int parallelism_depth = 6) -> bool {
  using Index0 = typename Policy0::index_type;
  using Index1 = typename Policy1::index_type;

  using buff0_t = decltype(tf::core::make_local_buffer_for_transformed(
      std::declval<const tf::form<Dims, Policy0> &>()[Index0(0)],
      std::declval<const tf::form<Dims, Policy0> &>().frame()));
  using buff1_t = decltype(tf::core::make_local_buffer_for_transformed(
      std::declval<const tf::form<Dims, Policy1> &>()[Index1(0)],
      std::declval<const tf::form<Dims, Policy1> &>().frame()));
  constexpr bool returns_bool = !std::is_same_v<
      decltype(std::declval<const F1 &>()(
          tf::tag_id(
              Index0(0),
              tf::transformed(
                  std::declval<const tf::form<Dims, Policy0> &>()[Index0(0)] |
                      tf::tag(std::declval<buff0_t &>()),
                  std::declval<const tf::form<Dims, Policy0> &>().frame())),
          tf::tag_id(
              Index1(0),
              tf::transformed(
                  std::declval<const tf::form<Dims, Policy1> &>()[Index1(0)] |
                      tf::tag(std::declval<buff1_t &>()),
                  std::declval<const tf::form<Dims, Policy1> &>().frame())))),
      void>;

  if constexpr (returns_bool) {
    std::atomic_bool flag{false};
    auto abort_f = [&flag] { return flag.load(); };
    auto apply_f = [&flag, primitive_apply](auto &&obj0, auto &&obj1) -> bool {
      if (primitive_apply(obj0, obj1)) {
        flag.store(true);
        return true;
      }
      return false;
    };
    return spatial::search(form0, form1, check_bvs, apply_f, abort_f,
                           parallelism_depth);
  } else {
    auto apply_f = [primitive_apply](auto &&obj0, auto &&obj1) -> bool {
      primitive_apply(obj0, obj1);
      return false;
    };
    auto abort_f = [] { return false; };
    return spatial::search(form0, form1, check_bvs, apply_f, abort_f,
                           parallelism_depth);
  }
}

} // namespace tf::spatial
