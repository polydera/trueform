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
#include "../core/polygons.hpp"
#include "./arrangement_config.hpp"
#include "./arrangement_graph.hpp"
#include "./operands/make_completed_graph.hpp"
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
  return tf::arrangement::dispatch::make_completed_graph<Int>(
      std::move(in_forms), config, {});
}

/// @ingroup arrangement
/// @brief Single-form overload: the graph is the form's self
///        arrangement (`within` is implied).
template <typename Int = tf::none_t, typename Policy>
auto make_arrangement_graph(const tf::polygons<Policy> &form,
                            tf::arrangement_config config = {}) {
  return tf::arrangement::dispatch::make_completed_graph<Int>(form, config, {});
}

/// @ingroup arrangement
/// @brief Two-form overload — the forms may be DIFFERENT types; the
///        pair policy erases the difference behind
///        `apply_to_form(tag, f)`.
template <typename Int = tf::none_t, typename Policy0, typename Policy1>
auto make_arrangement_graph(const tf::polygons<Policy0> &form0,
                            const tf::polygons<Policy1> &form1,
                            tf::arrangement_config config = {}) {
  return tf::arrangement::dispatch::make_completed_graph<Int>(form0, form1,
                                                              config, {});
}

} // namespace tf
