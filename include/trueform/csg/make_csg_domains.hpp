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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/none.hpp"
#include "../core/views/enumerate.hpp"
#include "../reindex/return_index_map.hpp"
#include "../reindex/return_source_ids.hpp"
#include "../topology/domain_config.hpp"
#include "./csg_graph.hpp"
#include "./expression/expr.hpp"
#include "./graph/compute_domain_membership.hpp"
#include "./graph/compute_domain_partition.hpp"
#include "./graph/make_csg_domains.hpp"
#include "./graph/structural_membership.hpp"
#include "./expression.hpp"
#include <type_traits>

namespace tf {

/// @cond INTERNAL
namespace detail {

/// Default flag set for @ref tf::make_csg_domains: drop the universe and
/// fuse open fragments into it.
inline constexpr tf::domain_config csg_domains_default_config =
    tf::domain_config::exclude_outer_shell |
    tf::domain_config::ignore_open_fragments;

/// Shared implementation: coarsen domains under `config`, filter by `E`,
/// then emit one watertight mesh per surviving coarse domain. `WantLabels`
/// additionally returns per-cell (tag, face) provenance blocks; `WantPointMap`
/// bundles those plus per-cell point provenance into a csg_domains_index_map.
template <typename OutputCoordinateType, bool WantLabels = false,
          bool WantPointMap = false, typename Forms, typename Structs,
          typename Int, typename Expr>
auto make_csg_domains_impl(const tf::csg_graph<Forms, Structs, Int> &graph,
                           Expr E, tf::domain_config config) {
  using InputReal =
      typename tf::csg_graph<Forms, Structs, Int>::input_real_type;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;

  using Index = typename tf::csg_graph<Forms, Structs, Int>::index_type;
  // Within-builds read structurally: parity bits misread the
  // double-covered pockets of a self arrangement, whether the graph is
  // one self-arranged form or N forms of which some self-overlap. The
  // volume argmin seeds the universe; the nesting merges applied inside
  // the membership's union-find lift it to the full universe CLASS (all
  // contact-free exteriors — disjoint or nested components), and every
  // coarse domain outside that class is an interior cell.
  Index universe_fine =
      graph.with_self()
          ? Index(tf::csg::graph::find_universe_domain(graph.domain_volumes()))
          : Index(-1);
  auto membership = tf::csg::graph::compute_domain_membership(
      graph.descriptor(), graph.inclusion(),
      graph.arrangement().open_component_mask(),
      graph.domain_nesting_merges(), config, E, universe_fine);
  auto part = tf::csg::graph::compute_domain_partition(
      membership.domain_of_side, membership.n_components, membership.keep);
  auto result =
      tf::csg::graph::make_csg_domains<RealOut, WantLabels, WantPointMap>(
          graph.arrangement(), graph.face_cuts(), graph.created_points(),
          graph.forms(), part, graph.converter(), graph.loop_triangulations());
  if constexpr (WantPointMap) {
    // ids[k] is the coarse domain id; unpack its representative's operand
    // bits into the cell-major inclusion matrix.
    const auto &ids = std::get<1>(result);
    auto &imap = std::get<2>(result);
    const std::size_t n_ops = graph.forms().size();
    const std::size_t words = membership.words_per_domain;
    imap.inclusion = tf::blocked_buffer<bool, tf::dynamic_size>(n_ops);
    imap.inclusion.allocate(ids.size());
    auto *cell_bits = imap.inclusion.data_buffer().data();
    tf::parallel_for_each(
        tf::enumerate(tf::make_range(ids)), [&](auto pair) {
          const auto &[k, id] = pair;
          const std::size_t base = static_cast<std::size_t>(id) * words;
          auto *row = cell_bits + std::size_t(k) * n_ops;
          for (std::size_t i = 0; i < n_ops; ++i)
            row[i] =
                ((membership.rep_bits[base + i / 32] >> (i % 32)) & 1u) != 0;
        });
  }
  return result;
}

} // namespace detail
/// @endcond

/// @ingroup csg
/// @brief Extract every kept 3D domain of an N-form CSG arrangement as its
///        OWN watertight mesh.
///
/// Domains are first coarsened according to `config`
/// (@ref tf::domain_config), then filtered by the boolean expression `e`,
/// then each surviving coarse domain is emitted as a separate
/// `polygons_buffer`. The same flags drive the materialized topology path
/// (@ref tf::make_domain_labels):
///   - `ignore_open_fragments` fuses the two sides of every open
///     (boundary-carrying) component — including sheet halves, which the
///     arrangement leaves un-merged — into a single domain. The merged
///     sheet halves join the universe, so they vanish from the output.
///   - `exclude_outer_shell` drops the unbounded universe (any domain
///     with an all-zero inclusion bitvector). With this flag off, the
///     outside is recoverable via an expression such as
///     `~tf::csg::op(0) & ~tf::csg::op(1)`.
///
/// `config` defaults to `exclude_outer_shell | ignore_open_fragments`.
///
/// @tparam OutputCoordinateType Output coordinate type. Defaults to
///         @c tf::none_t, resolving to the input forms' coordinate type.
/// @return `{ cells, ids }`: one mesh per kept domain; `ids[k]` is the
///         COARSE domain id of cell `k`.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      const tf::csg::expr &e, tf::domain_config config) {
  auto E = e.compile().evaluator();
  return detail::make_csg_domains_impl<OutputCoordinateType>(graph, E, config);
}

/// @ingroup csg
/// @brief Filter overload with the default flag set
///        (`exclude_outer_shell | ignore_open_fragments`).
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      const tf::csg::expr &e) {
  return make_csg_domains<OutputCoordinateType>(
      graph, e, detail::csg_domains_default_config);
}

/// @ingroup csg
/// @brief Select-all overload, explicit `config`.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      tf::domain_config config) {
  return detail::make_csg_domains_impl<OutputCoordinateType>(
      graph, [](const auto &) { return true; }, config);
}

/// @ingroup csg
/// @brief Select-all overload with the default flag set.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph) {
  return make_csg_domains<OutputCoordinateType>(
      graph, detail::csg_domains_default_config);
}

// ---- return_source_ids overloads: additionally return per-cell provenance
// as two offset_block_buffers parallel to `cells` --------------------------
//   tag_blocks[k][j]  = the input form of cell k's face j
//   face_blocks[k][j] = the original face id within that form
// Returns (cells, ids, tag_blocks, face_blocks).

/// @ingroup csg
/// @brief Filter overload with explicit `config`, returning per-cell
///        (tag, face) provenance.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      const tf::csg::expr &e, tf::domain_config config,
                      tf::return_source_ids_t) {
  auto E = e.compile().evaluator();
  return detail::make_csg_domains_impl<OutputCoordinateType, true>(graph, E,
                                                                   config);
}

/// @ingroup csg
/// @brief Filter overload (default flags), returning per-cell provenance.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      const tf::csg::expr &e, tf::return_source_ids_t) {
  return make_csg_domains<OutputCoordinateType>(
      graph, e, detail::csg_domains_default_config, tf::return_source_ids);
}

/// @ingroup csg
/// @brief Select-all overload with explicit `config`, returning per-cell
///        provenance.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      tf::domain_config config, tf::return_source_ids_t) {
  return detail::make_csg_domains_impl<OutputCoordinateType, true>(
      graph, [](const auto &) { return true; }, config);
}

/// @ingroup csg
/// @brief Select-all overload (default flags), returning per-cell provenance.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      tf::return_source_ids_t) {
  return make_csg_domains<OutputCoordinateType>(
      graph, detail::csg_domains_default_config, tf::return_source_ids);
}

// ---- return_index_map overloads: bundle the per-cell (tag, face) and
// (tag, point) provenance into a tf::csg_domains_index_map ------------------
// Returns (cells, ids, index_map).

/// @ingroup csg
/// @brief Filter overload with explicit `config`, returning a
///        @ref tf::csg_domains_index_map.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      const tf::csg::expr &e, tf::domain_config config,
                      tf::return_index_map_t) {
  auto E = e.compile().evaluator();
  return detail::make_csg_domains_impl<OutputCoordinateType, true, true>(
      graph, E, config);
}

/// @ingroup csg
/// @brief Filter overload (default flags), returning a
///        @ref tf::csg_domains_index_map.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      const tf::csg::expr &e, tf::return_index_map_t) {
  return make_csg_domains<OutputCoordinateType>(
      graph, e, detail::csg_domains_default_config, tf::return_index_map);
}

/// @ingroup csg
/// @brief Select-all overload with explicit `config`, returning a
///        @ref tf::csg_domains_index_map.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      tf::domain_config config, tf::return_index_map_t) {
  return detail::make_csg_domains_impl<OutputCoordinateType, true, true>(
      graph, [](const auto &) { return true; }, config);
}

/// @ingroup csg
/// @brief Select-all overload (default flags), returning a
///        @ref tf::csg_domains_index_map.
template <typename OutputCoordinateType = tf::none_t, typename Forms,
          typename Structs, typename Int>
auto make_csg_domains(const tf::csg_graph<Forms, Structs, Int> &graph,
                      tf::return_index_map_t) {
  return make_csg_domains<OutputCoordinateType>(
      graph, detail::csg_domains_default_config, tf::return_index_map);
}

} // namespace tf
