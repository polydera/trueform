/*
 * Copyright (c) 2026 XLAB
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
#include "../core/none.hpp"
#include "../core/range.hpp"
#include "../core/small_vector.hpp"
#include "./arrangement_config.hpp"
#include "./arrangement_graph.hpp"
#include "./operands/boolean.hpp"
#include "./operands/make_graph.hpp"
#include "./policy/arrangement_pair_policy.hpp"
#include "./policy/arrangement_range_policy.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/task_group.h"
#include <array>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup arrangement
/// @brief Build a @ref tf::arrangement_graph for `forms`, auto-tagging
///        any missing structures (manifold-edge link, face membership,
///        AABB tree) in parallel. A frame is never built — it states a
///        transformation only the caller knows, so tag it yourself.
///
/// Returns by value, owning every structure it built. The operands
/// themselves follow the storage policy: a form passed on its own or as a
/// pair is copied in, while a *range* of forms is stored as the view it is,
/// so the graph must not outlive the container behind it.
///
/// The arrangement is classification-free — for boolean expressions
/// and domain reads build a @ref tf::csg_graph instead.
///
/// @tparam Int  Exact-integer override (defaulted to @c tf::none_t →
///              resolved from the input coordinate type).
template <typename Int = tf::none_t, typename Forms>
auto make_arrangement_graph(Forms in_forms,
                            tf::arrangement_config config = {}) {
  auto forms = tf::make_range(in_forms);
  using S =
      decltype(tf::arrangement::dispatch::make_missing_structures(forms[0]));

  if constexpr (std::is_same_v<S, tf::none_t>) {
    return tf::arrangement::dispatch::make_graph<Int>(
        tf::arrangement::arrangement_range_policy<Forms, S>(
            std::move(forms), tf::small_vector<S, 10>{}),
        config);
  } else {
    const auto n = forms.size();
    tf::small_vector<S, 10> structs(n);
    tbb::task_group tg;
    for (std::size_t i = 0; i < n; ++i)
      tg.run([&, i] {
        structs[i] =
            tf::arrangement::dispatch::make_missing_structures(forms[i]);
      });
    tg.wait();
    return tf::arrangement::dispatch::make_graph<Int>(
        tf::arrangement::arrangement_range_policy<Forms, S>(std::move(forms),
                                                            std::move(structs)),
        config);
  }
}

/// @ingroup arrangement
/// @brief Single-form overload: the graph is the form's self
///        arrangement (@ref tf::intersect_mode::within is implied).
template <typename Int = tf::none_t, typename Policy>
auto make_arrangement_graph(const tf::polygons<Policy> &form,
                            tf::arrangement_config config = {}) {
  using S = decltype(tf::arrangement::dispatch::make_missing_structures(form));
  std::array<tf::polygons<Policy>, 1> forms = {form};
  if constexpr (std::is_same_v<S, tf::none_t>) {
    using P = tf::arrangement::arrangement_range_policy<
        std::array<tf::polygons<Policy>, 1>, S>;
    return tf::arrangement::dispatch::make_graph<Int>(
        P(std::move(forms), tf::small_vector<S, 10>{}), config);
  } else {
    tf::small_vector<S, 10> structs(1);
    structs[0] = tf::arrangement::dispatch::make_missing_structures(forms[0]);
    using P = tf::arrangement::arrangement_range_policy<
        std::array<tf::polygons<Policy>, 1>, S>;
    return tf::arrangement::dispatch::make_graph<Int>(
        P(std::move(forms), std::move(structs)), config);
  }
}

/// @ingroup arrangement
/// @brief Two-form overload — the forms may be DIFFERENT types; the
///        pair policy erases the difference behind
///        `apply_to_form(tag, f)`.
template <typename Int = tf::none_t, typename Policy0, typename Policy1>
auto make_arrangement_graph(const tf::polygons<Policy0> &form0,
                            const tf::polygons<Policy1> &form1,
                            tf::arrangement_config config = {}) {
  using S0 =
      decltype(tf::arrangement::dispatch::make_missing_structures(form0));
  using S1 =
      decltype(tf::arrangement::dispatch::make_missing_structures(form1));
  using P = tf::arrangement::arrangement_pair_policy<tf::polygons<Policy0>, S0,
                                             tf::polygons<Policy1>, S1>;
  if constexpr (std::is_same_v<S0, tf::none_t> &&
                std::is_same_v<S1, tf::none_t>) {
    return tf::arrangement::dispatch::make_graph<Int>(
        P(form0, tf::none, form1, tf::none), config);
  } else if constexpr (!std::is_same_v<S0, tf::none_t> &&
                       !std::is_same_v<S1, tf::none_t>) {
    S0 s0;
    S1 s1;
    tbb::parallel_invoke(
        [&] { s0 = tf::arrangement::dispatch::make_missing_structures(form0); },
        [&] {
          s1 = tf::arrangement::dispatch::make_missing_structures(form1);
        });
    return tf::arrangement::dispatch::make_graph<Int>(
        P(form0, std::move(s0), form1, std::move(s1)), config);
  } else if constexpr (!std::is_same_v<S0, tf::none_t>) {
    auto s0 = tf::arrangement::dispatch::make_missing_structures(form0);
    return tf::arrangement::dispatch::make_graph<Int>(
        P(form0, std::move(s0), form1, tf::none), config);
  } else {
    auto s1 = tf::arrangement::dispatch::make_missing_structures(form1);
    return tf::arrangement::dispatch::make_graph<Int>(
        P(form0, tf::none, form1, std::move(s1)), config);
  }
}

} // namespace tf
