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
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/buffer.hpp"
#include "../core/none.hpp"
#include "../core/range.hpp"
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
/// `sheets` lists form indices to treat as sheets: oriented separators
/// rather than volumes. A sheet's fragments always divide their two
/// sides (no Mode-2 self-merge), and its operand bit means "on the
/// back side of the sheet's normal" — so `difference`/`intersection`
/// cut volumes against it into closed, capped halves. Declaring a
/// clean, outward-oriented closed mesh a sheet is a no-op; forms not
/// listed keep volume semantics, with open fragments (fins, damage)
/// fused away as before.
///
/// @tparam Int  Exact-integer override (defaulted to @c tf::none_t →
///              resolved from the input coordinate type).
///
/// @code{.cpp}
/// auto graph = tf::make_csg_graph(forms);
/// auto m_diff  = tf::make_csg_mesh(graph, tf::csg::difference(0, 1));
/// auto m_union = tf::make_csg_mesh(graph, tf::csg::merge(0, 1));
///
/// std::array<int, 1> sheets = {1};
/// auto cutter = tf::make_csg_graph(forms, tf::make_range(sheets));
/// auto upper  = tf::make_csg_mesh(cutter, tf::csg::difference(0, 1));
/// @endcode
template <typename Int = tf::none_t, typename Forms, typename Iter,
          std::size_t N>
auto make_csg_graph(Forms in_forms, tf::range<Iter, N> sheets,
                     tf::intersect_config config =
                         {tf::intersect_mode::primitives |
                          tf::intersect_mode::resolve_crossing_contours},
                     tf::triangulation_type tri =
                         tf::triangulation_type::cdt) {
  auto forms = tf::make_range(in_forms);
  using S = decltype(tf::cut::dispatch::make_missing_structures(forms[0]));

  tf::buffer<char> is_sheet;
  if (sheets.size() > 0) {
    is_sheet.allocate(forms.size());
    tf::parallel_fill(is_sheet, char(0));
    for (auto id : sheets)
      is_sheet[static_cast<std::size_t>(id)] = char(1);
  }

  if constexpr (std::is_same_v<S, tf::none_t>) {
    return tf::csg_graph<Forms, S, Int>(std::move(forms),
                                          tf::small_vector<S, 10>{}, config,
                                          std::move(is_sheet), tri);
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
                                          std::move(structs), config,
                                          std::move(is_sheet), tri);
  }
}

/// @ingroup csg
/// @brief No-sheets overload: every form is a volume.
template <typename Int = tf::none_t, typename Forms>
auto make_csg_graph(Forms in_forms,
                     tf::intersect_config config =
                         {tf::intersect_mode::primitives |
                          tf::intersect_mode::resolve_crossing_contours},
                     tf::triangulation_type tri =
                         tf::triangulation_type::cdt) {
  const int *none = nullptr;
  return make_csg_graph<Int>(std::move(in_forms), tf::make_range(none, none),
                             config, tri);
}

/// @ingroup csg
/// @brief Triangulation-only overload: default intersect config.
template <typename Int = tf::none_t, typename Forms, typename Iter,
          std::size_t N>
auto make_csg_graph(Forms in_forms, tf::range<Iter, N> sheets,
                     tf::triangulation_type tri) {
  return make_csg_graph<Int>(std::move(in_forms), sheets,
                             {tf::intersect_mode::primitives |
                              tf::intersect_mode::resolve_crossing_contours},
                             tri);
}

/// @ingroup csg
/// @brief Triangulation-only overload: no sheets, default intersect config.
template <typename Int = tf::none_t, typename Forms>
auto make_csg_graph(Forms in_forms, tf::triangulation_type tri) {
  return make_csg_graph<Int>(std::move(in_forms),
                             {tf::intersect_mode::primitives |
                              tf::intersect_mode::resolve_crossing_contours},
                             tri);
}

} // namespace tf
