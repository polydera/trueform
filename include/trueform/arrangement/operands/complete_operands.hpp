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
#include "../../core/none.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../policy/arrangement_pair_policy.hpp"
#include "../policy/arrangement_range_policy.hpp"
#include "./boolean.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/task_group.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// The operands completed: the storage policy an arrangement is built on,
/// holding each operand with every tree, face membership and manifold edge
/// link it lacked, built only where missing and in parallel, so a fully
/// tagged call site completes with zero structure work. A frame is kept as
/// tagged. Every converter and every graph is built on what this returns.
template <typename Forms> auto complete_operands(Forms in_forms) {
  auto forms = tf::make_range(in_forms);
  using S =
      decltype(tf::arrangement::dispatch::make_missing_structures(forms[0]));

  if constexpr (std::is_same_v<S, tf::none_t>) {
    return tf::arrangement::arrangement_range_policy<Forms, S>(
        std::move(forms), tf::small_vector<S, 10>{});
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
    return tf::arrangement::arrangement_range_policy<Forms, S>(
        std::move(forms), std::move(structs));
  }
}

/// @overload The one-form shape: the form's self arrangement.
template <typename Policy>
auto complete_operands(const tf::polygons<Policy> &form) {
  using S = decltype(tf::arrangement::dispatch::make_missing_structures(form));
  std::array<tf::polygons<Policy>, 1> forms = {form};
  using P = tf::arrangement::arrangement_range_policy<
      std::array<tf::polygons<Policy>, 1>, S>;
  if constexpr (std::is_same_v<S, tf::none_t>) {
    return P(std::move(forms), tf::small_vector<S, 10>{});
  } else {
    tf::small_vector<S, 10> structs(1);
    structs[0] = tf::arrangement::dispatch::make_missing_structures(forms[0]);
    return P(std::move(forms), std::move(structs));
  }
}

/// @overload The two-form shape: the forms may be DIFFERENT types, erased
/// behind the pair policy's `apply_to_form(tag, f)`.
template <typename Policy0, typename Policy1>
auto complete_operands(const tf::polygons<Policy0> &form0,
                       const tf::polygons<Policy1> &form1) {
  using S0 =
      decltype(tf::arrangement::dispatch::make_missing_structures(form0));
  using S1 =
      decltype(tf::arrangement::dispatch::make_missing_structures(form1));
  using P = tf::arrangement::arrangement_pair_policy<tf::polygons<Policy0>, S0,
                                                     tf::polygons<Policy1>, S1>;
  if constexpr (std::is_same_v<S0, tf::none_t> &&
                std::is_same_v<S1, tf::none_t>) {
    return P(form0, tf::none, form1, tf::none);
  } else if constexpr (!std::is_same_v<S0, tf::none_t> &&
                       !std::is_same_v<S1, tf::none_t>) {
    S0 s0;
    S1 s1;
    tbb::parallel_invoke(
        [&] { s0 = tf::arrangement::dispatch::make_missing_structures(form0); },
        [&] {
          s1 = tf::arrangement::dispatch::make_missing_structures(form1);
        });
    return P(form0, std::move(s0), form1, std::move(s1));
  } else if constexpr (!std::is_same_v<S0, tf::none_t>) {
    auto s0 = tf::arrangement::dispatch::make_missing_structures(form0);
    return P(form0, std::move(s0), form1, tf::none);
  } else {
    auto s1 = tf::arrangement::dispatch::make_missing_structures(form1);
    return P(form0, tf::none, form1, std::move(s1));
  }
}

} // namespace tf::arrangement
