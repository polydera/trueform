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
#include "../arrangement_config.hpp"
#include "../identity/near_identity_request.hpp"
#include "../policy/arrangement_pair_policy.hpp"
#include "../policy/arrangement_range_policy.hpp"
#include "./boolean.hpp"
#include "./make_graph.hpp"
#include "tbb/parallel_invoke.h"
#include "tbb/task_group.h"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::arrangement::dispatch {

/// The operands completed and the graph built (@ref make_graph): each shape
/// builds only the structures it is missing, in parallel, so a fully tagged
/// call site constructs with zero structure work. `request` is what a tier
/// above asks of the build; the public entry asks nothing.
template <typename Int, typename Forms>
auto make_completed_graph(Forms in_forms, tf::arrangement_config config,
                          tf::arrangement::near_identity_request request) {
  auto forms = tf::make_range(in_forms);
  using S = decltype(make_missing_structures(forms[0]));

  if constexpr (std::is_same_v<S, tf::none_t>) {
    return make_graph<Int>(tf::arrangement::arrangement_range_policy<Forms, S>(
                               std::move(forms), tf::small_vector<S, 10>{}),
                           config, request);
  } else {
    const auto n = forms.size();
    tf::small_vector<S, 10> structs(n);
    tbb::task_group tg;
    for (std::size_t i = 0; i < n; ++i)
      tg.run([&, i] { structs[i] = make_missing_structures(forms[i]); });
    tg.wait();
    return make_graph<Int>(tf::arrangement::arrangement_range_policy<Forms, S>(
                               std::move(forms), std::move(structs)),
                           config, request);
  }
}

/// The one-form shape: the form's self arrangement.
template <typename Int, typename Policy>
auto make_completed_graph(const tf::polygons<Policy> &form,
                          tf::arrangement_config config,
                          tf::arrangement::near_identity_request request) {
  using S = decltype(make_missing_structures(form));
  std::array<tf::polygons<Policy>, 1> forms = {form};
  using P = tf::arrangement::arrangement_range_policy<
      std::array<tf::polygons<Policy>, 1>, S>;
  if constexpr (std::is_same_v<S, tf::none_t>) {
    return make_graph<Int>(P(std::move(forms), tf::small_vector<S, 10>{}),
                           config, request);
  } else {
    tf::small_vector<S, 10> structs(1);
    structs[0] = make_missing_structures(forms[0]);
    return make_graph<Int>(P(std::move(forms), std::move(structs)), config,
                           request);
  }
}

/// The two-form shape: the forms may be DIFFERENT types, erased behind the
/// pair policy's `apply_to_form(tag, f)`.
template <typename Int, typename Policy0, typename Policy1>
auto make_completed_graph(const tf::polygons<Policy0> &form0,
                          const tf::polygons<Policy1> &form1,
                          tf::arrangement_config config,
                          tf::arrangement::near_identity_request request) {
  using S0 = decltype(make_missing_structures(form0));
  using S1 = decltype(make_missing_structures(form1));
  using P = tf::arrangement::arrangement_pair_policy<tf::polygons<Policy0>, S0,
                                                     tf::polygons<Policy1>, S1>;
  if constexpr (std::is_same_v<S0, tf::none_t> &&
                std::is_same_v<S1, tf::none_t>) {
    return make_graph<Int>(P(form0, tf::none, form1, tf::none), config,
                           request);
  } else if constexpr (!std::is_same_v<S0, tf::none_t> &&
                       !std::is_same_v<S1, tf::none_t>) {
    S0 s0;
    S1 s1;
    tbb::parallel_invoke([&] { s0 = make_missing_structures(form0); },
                         [&] { s1 = make_missing_structures(form1); });
    return make_graph<Int>(P(form0, std::move(s0), form1, std::move(s1)),
                           config, request);
  } else if constexpr (!std::is_same_v<S0, tf::none_t>) {
    auto s0 = make_missing_structures(form0);
    return make_graph<Int>(P(form0, std::move(s0), form1, tf::none), config,
                           request);
  } else {
    auto s1 = make_missing_structures(form1);
    return make_graph<Int>(P(form0, tf::none, form1, std::move(s1)), config,
                           request);
  }
}

} // namespace tf::arrangement::dispatch
