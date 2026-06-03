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
#include "../core/small_vector.hpp"
#include "../cut/dispatch/boolean.hpp"
#include "../intersect/intersect_config.hpp"
#include "./csg_graph.hpp"
#include "tbb/task_group.h"
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup csg
/// @brief Build a @ref tf::csg_graph for `forms`, auto-tagging any
///        missing structures (manifold-edge link, face membership,
///        AABB tree, frame) in parallel. Returns by value; the graph
///        owns everything it needs to extract meshes.
///
/// @tparam Int  Exact-integer override (defaulted to @c tf::none_t →
///              resolved from the input coordinate type).
///
/// @code{.cpp}
/// auto graph = tf::make_csg_graph(forms);
/// auto m_diff  = tf::make_csg_mesh(graph, tf::csg::difference(0, 1));
/// auto m_union = tf::make_csg_mesh(graph, tf::csg::merge(0, 1));
/// @endcode
template <typename Int = tf::none_t, typename Forms>
auto make_csg_graph(Forms in_forms,
                     tf::intersect_config config =
                         {tf::intersect_mode::primitives |
                          tf::intersect_mode::resolve_crossing_contours}) {
  auto forms = tf::make_range(in_forms);
  using S = decltype(tf::cut::dispatch::make_missing_structures(forms[0]));

  if constexpr (std::is_same_v<S, tf::none_t>) {
    return tf::csg_graph<Forms, S, Int>(std::move(forms),
                                          tf::small_vector<S, 10>{}, config);
  } else {
    const auto n = forms.size();
    tf::small_vector<S, 10> structs(n);
    tbb::task_group tg;
    for (std::size_t i = 0; i < n; ++i)
      tg.run([&, i] {
        structs[i] = tf::cut::dispatch::make_missing_structures(forms[i]);
      });
    tg.wait();
    return tf::csg_graph<Forms, S, Int>(std::move(forms),
                                          std::move(structs), config);
  }
}

} // namespace tf
