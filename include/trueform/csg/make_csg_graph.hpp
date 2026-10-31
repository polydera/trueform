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
#include "../arrangement/identity/near_identity_request.hpp"
#include "../arrangement/operands/make_completed_graph.hpp"
#include "../core/buffer.hpp"
#include "../core/config/verify.hpp"
#include "../core/none.hpp"
#include "../core/range.hpp"
#include "./csg_graph.hpp"
#include "./graph/assert_domains_partition_space.hpp"
#include "./graph/assert_faces_are_named.hpp"
#include "./graph/assert_nesting_merges_are_contained.hpp"
#include "./graph/assert_nesting_merges_keep_closed_bits.hpp"
#include "./graph/assert_radial_rings_are_even.hpp"
#include "./graph/census_conviction.hpp"
#include "./graph/census_statement.hpp"
#include "./graph/collect_census_id_pairs.hpp"
#include "./graph/make_census_classes.hpp"
#include "./graph/make_closed_volume_tags.hpp"
#include "./graph/make_sheet_mask.hpp"
#include <array>
#include <cstddef>
#include <type_traits>
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

/// The census rounds a build may run. Measured fixed points stand at two
/// rounds; the cap stops a wave that would not end.
inline constexpr std::size_t census_round_cap = 8;

/// The near table the census asks of the arrangement it classifies: one
/// input quantum, the census's own weld radius.
inline constexpr tf::arrangement::near_identity_request census_near_request{1};

/// The classification of an arrangement, run to the census's fixed point.
/// While the census convicts the graph (@ref
/// tf::csg::graph::make_census_conviction), the identities it may fuse
/// inside its convictions are stated against the standing arrangement
/// (@ref tf::arrangement_graph::state_census_diff). A green graph pays the
/// conviction's two parallel passes and nothing else; an integral input has
/// no quantum, so its near table is empty and its sites state nothing to
/// fuse.
///
/// The rounds stop on a verdict: the graph is green, the carve refused, or
/// a round has nothing left to state — no welds, and no carve or the one
/// escalation already spent. A carve therefore enters at most once and a
/// refused carve is never stated again. A round of welds carries no verdict
/// of its own, so a weld statement that regenerates itself runs until @ref
/// census_round_cap stops it. A run that stops convicted publishes its
/// rounds, its statements, its standing crossings and the conviction it
/// stands under in `census_record`, and whether the cap stopped it.
template <typename Int, template <typename, typename> class Arrangement,
          typename Policy, typename Iter, std::size_t N>
auto census_graph_over(Arrangement<Policy, Int> arr,
                       tf::range<Iter, N> sheets) {
  // the classification is certified where it is claimed: a green conviction
  // says every admitted ring pairs its domains, every nesting merge unites
  // one region, the walls close every region they bound and every face
  // stands in one, and the census is the one authority on a graph that
  // is not yet, or cannot be, repaired
  const auto certify = [](const auto &graph) {
    if constexpr (tf::core::verification) {
      using graph_t = std::decay_t<decltype(graph)>;
      const auto &arrangement = graph.arrangement();
      const auto &desc = graph.descriptor();
      tf::csg::graph::assert_radial_rings_are_even(
          graph.labels().triangle_labels(), desc.fans,
          arrangement.exposed_of_row(), desc.valid, desc.domain_of_side);
      const auto closed_volume_tags = tf::csg::graph::make_closed_volume_tags(
          desc, graph.labels(), arrangement.n_tags(), graph.is_sheet());
      tf::csg::graph::assert_nesting_merges_keep_closed_bits(
          graph.domain_nesting_merges(), graph.inclusion(),
          closed_volume_tags);
      const auto apply_to_form = arrangement.apply_to_form();
      const auto read_point = arrangement.lattice().reader(apply_to_form);
      tf::csg::graph::assert_nesting_merges_are_contained<
          typename graph_t::index_type, typename graph_t::resolved_int_type>(
          graph.domain_nesting_merges(), graph.inclusion(), closed_volume_tags,
          desc, graph.domain_volumes(), arrangement, graph.labels(),
          apply_to_form, read_point);
      tf::csg::graph::assert_domains_partition_space<
          typename graph_t::index_type, typename graph_t::resolved_int_type>(
          desc, arrangement, graph.labels(), graph.crossings(),
          graph.is_sheet(), apply_to_form, read_point);
      tf::csg::graph::assert_faces_are_named<
          typename graph_t::index_type, typename graph_t::resolved_int_type>(
          arrangement, graph.labels(), apply_to_form, read_point);
    }
  };
  auto g = csg_graph_over<Int>(std::move(arr), sheets);
  auto conviction = tf::csg::graph::make_census_conviction(g);
  const auto record = [&conviction](std::size_t rounds, std::size_t stated,
                                    bool capped) {
    return
        typename std::decay_t<decltype(g)>::census_record{rounds,
                                                          stated,
                                                          conviction.crossed,
                                                          conviction.failed,
                                                          conviction.odd,
                                                          conviction.self_sided,
                                                          capped};
  };
  if (conviction.green()) {
    certify(g);
    g.state_census(record(0, 0, false));
    return g;
  }
  using resolved_int_type =
      typename Arrangement<Policy, Int>::resolved_int_type;
  using graph_index_type = typename std::decay_t<decltype(g)>::index_type;
  tf::buffer<std::array<graph_index_type, 4>> pairs;
  std::size_t stated = 0;
  std::size_t rounds = 0;
  bool escalated = false;
  bool verdict = false;
  for (std::size_t round = 0; round < census_round_cap; ++round) {
    // a round renumbers the ids, so the pairs are this graph's alone
    tf::csg::graph::collect_census_id_pairs(g, conviction, pairs);
    const auto classes = tf::csg::graph::make_census_classes(pairs);
    const auto statement =
        tf::csg::graph::make_census_statement(g, classes, pairs);
    // the welds state first and the carve escalates: an original's
    // retirement is cut into the standing world only when the welds'
    // fixed point still stands convicted, and the plane-tier welds onto its
    // mint enter with it
    const auto welds = statement.la_welds.size() + statement.pa_welds.size();
    const bool fresh = welds != 0;
    if (!fresh && (escalated || statement.carve_relations.size() == 0)) {
      verdict = true;
      break;
    }
    stated += fresh ? welds
                    : statement.carve_relations.size() +
                          statement.carve_pa_welds.size();
    const tf::buffer<std::array<graph_index_type, 3>> no_rows;
    const tf::buffer<tf::point<resolved_int_type, 3>> no_mints;
    auto arr = std::move(g).release_arrangement();
    bool fused;
    if (fresh)
      fused = arr.state_census_diff(statement.la_welds, no_mints, no_rows,
                                    statement.pa_welds);
    else {
      fused = arr.state_census_diff(no_rows, statement.minted,
                                    statement.carve_relations,
                                    statement.carve_pa_welds);
      escalated = true;
    }
    ++rounds;
    g = csg_graph_over<Int>(std::move(arr), sheets);
    conviction = tf::csg::graph::make_census_conviction(g);
    if (conviction.green()) {
      certify(g);
      verdict = true;
      break;
    }
    // a refused carve cannot advance the fixed point: the same relations
    // would only be collected again
    if (!fused) {
      verdict = true;
      break;
    }
  }
  g.state_census(record(rounds, stated, !verdict));
  return g;
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
  return csg::census_graph_over<Int>(
      tf::arrangement::dispatch::make_completed_graph<Int>(
          std::move(in_forms), config, csg::census_near_request),
      sheets);
}

/// @ingroup csg
/// @brief No-sheets overload: every form is a volume.
template <typename Int = tf::none_t, typename Forms>
auto make_csg_graph(Forms in_forms, tf::arrangement_config config = {}) {
  return make_csg_graph<Int>(std::move(in_forms), csg::no_sheets(), config);
}

/// @ingroup csg
/// @brief Single-form overload: the graph is the form's self
///        arrangement (`within` is implied).
///        The structural reads apply — @ref tf::make_outer_shell and
///        @ref tf::make_csg_domains; boolean expressions need two
///        operands.
template <typename Int = tf::none_t, typename Policy>
auto make_csg_graph(const tf::polygons<Policy> &form,
                    tf::arrangement_config config = {}) {
  return csg::census_graph_over<Int>(
      tf::arrangement::dispatch::make_completed_graph<Int>(
          form, config, csg::census_near_request),
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
  return csg::census_graph_over<Int>(
      tf::arrangement::dispatch::make_completed_graph<Int>(
          form0, form1, config, csg::census_near_request),
      sheets);
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
