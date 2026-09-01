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
#include "../arrangement/arrangement_config.hpp"
#include "../arrangement/make_arrangement_graph.hpp"
#include "../core/none.hpp"
#include "../core/range.hpp"
#include "./csg_graph.hpp"
#include "./graph/make_sheet_mask.hpp"
#include <utility>

namespace tf {

/// @cond INTERNAL
namespace csg {

/// The classification tier over an arrangement — the only thing
/// @ref tf::make_csg_graph adds. Which operand shape produced the
/// arrangement is already settled by then.
template <typename Int, template <typename, typename> class Arrangement,
          typename Policy, typename Iter, std::size_t N>
auto csg_graph_over(Arrangement<Policy, Int> arr, tf::range<Iter, N> sheets) {
  auto mask = tf::csg::graph::make_sheet_mask(sheets, arr.n_tags());
  return tf::csg_graph<Policy, Int, Arrangement>(std::move(arr),
                                                 std::move(mask));
}

inline auto no_sheets() {
  const int *none = nullptr;
  return tf::make_range(none, none);
}

} // namespace csg
/// @endcond

/// @ingroup csg
/// @brief Build a @ref tf::csg_graph for `forms` — the arrangement of
///        @ref tf::make_arrangement_graph plus the classification tier
///        (descriptor, domain inclusions, volumes, sheet anchoring).
///
/// Structure completion, operand-shape dispatch and the arrangement
/// build all belong to @ref tf::make_arrangement_graph; this adds
/// classification and the sheet declaration, nothing else. Returns by
/// value; the graph owns everything it needs to extract meshes.
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
                    tf::arrangement_config config = {}) {
  return csg::csg_graph_over<Int>(
      tf::make_arrangement_graph<Int>(std::move(in_forms), config), sheets);
}

/// @ingroup csg
/// @brief No-sheets overload: every form is a volume.
template <typename Int = tf::none_t, typename Forms>
auto make_csg_graph(Forms in_forms, tf::arrangement_config config = {}) {
  return make_csg_graph<Int>(std::move(in_forms), csg::no_sheets(), config);
}

/// @ingroup csg
/// @brief Single-form overload: the graph is the form's self
///        arrangement (@ref tf::intersect_mode::within is implied).
///        The structural reads apply — @ref tf::make_outer_shell and
///        @ref tf::make_csg_domains; boolean expressions need two
///        operands.
template <typename Int = tf::none_t, typename Policy>
auto make_csg_graph(const tf::polygons<Policy> &form,
                    tf::arrangement_config config = {}) {
  return csg::csg_graph_over<Int>(tf::make_arrangement_graph<Int>(form, config),
                                  csg::no_sheets());
}

/// @ingroup csg
/// @brief Two-form overload — the forms may be DIFFERENT types; the
///        pair policy erases the difference behind
///        `apply_to_form(tag, f)`, so operand 0 is `form0` and operand
///        1 is `form1` in every expression built against this graph.
template <typename Int = tf::none_t, typename Policy0, typename Policy1,
          typename Iter, std::size_t N>
auto make_csg_graph(const tf::polygons<Policy0> &form0,
                    const tf::polygons<Policy1> &form1,
                    tf::range<Iter, N> sheets,
                    tf::arrangement_config config = {}) {
  return csg::csg_graph_over<Int>(
      tf::make_arrangement_graph<Int>(form0, form1, config), sheets);
}

/// @ingroup csg
/// @brief Two-form overload, both operands volumes.
template <typename Int = tf::none_t, typename Policy0, typename Policy1>
auto make_csg_graph(const tf::polygons<Policy0> &form0,
                    const tf::polygons<Policy1> &form1,
                    tf::arrangement_config config = {}) {
  return make_csg_graph<Int>(form0, form1, csg::no_sheets(), config);
}

} // namespace tf
