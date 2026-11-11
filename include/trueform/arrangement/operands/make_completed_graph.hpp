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
#include "../arrangement_config.hpp"
#include "../identity/near_identity_request.hpp"
#include "./complete_operands.hpp"
#include "./make_graph.hpp"
#include <utility>

namespace tf::arrangement::dispatch {

/// The operands completed (@ref tf::arrangement::complete_operands), the
/// house converter built over them, and the graph built on it
/// (@ref make_graph). `request` is what a tier above asks of the build; the
/// public entry asks nothing.
template <typename Int, typename Forms>
auto make_completed_graph(Forms in_forms, tf::arrangement_config config,
                          tf::arrangement::near_identity_request request) {
  return make_graph<Int>(
      tf::arrangement::complete_operands(std::move(in_forms)), tf::none, config,
      request);
}

/// @overload The one-form shape: the form's self arrangement.
template <typename Int, typename Policy>
auto make_completed_graph(const tf::polygons<Policy> &form,
                          tf::arrangement_config config,
                          tf::arrangement::near_identity_request request) {
  return make_graph<Int>(tf::arrangement::complete_operands(form), tf::none,
                         config, request);
}

/// @overload The two-form shape.
template <typename Int, typename Policy0, typename Policy1>
auto make_completed_graph(const tf::polygons<Policy0> &form0,
                          const tf::polygons<Policy1> &form1,
                          tf::arrangement_config config,
                          tf::arrangement::near_identity_request request) {
  return make_graph<Int>(tf::arrangement::complete_operands(form0, form1),
                         tf::none, config, request);
}

} // namespace tf::arrangement::dispatch
