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
#include "../topology/domain_config.hpp"
#include "./csg_graph.hpp"
#include "./expression/expr.hpp"
#include "./graph/compute_domain_membership.hpp"
#include "./graph/compute_domain_partition.hpp"
#include "./graph/make_csg_domains.hpp"
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
/// then emit one watertight mesh per surviving coarse domain.
template <typename OutputCoordinateType, typename Forms, typename Structs,
          typename Int, typename Expr>
auto make_csg_domains_impl(const tf::csg_graph<Forms, Structs, Int> &graph,
                           Expr E, tf::domain_config config) {
  using InputReal =
      typename tf::csg_graph<Forms, Structs, Int>::input_real_type;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;

  auto membership = tf::csg::graph::compute_domain_membership(
      graph.descriptor(), graph.inclusion(),
      graph.arrangement().open_component_mask(),
      graph.domain_nesting_merges(), config, E);
  auto part = tf::csg::graph::compute_domain_partition(
      membership.domain_of_side, membership.n_components, membership.keep);
  return tf::csg::graph::make_csg_domains<RealOut>(
      graph.arrangement(), graph.face_cuts(), graph.intersection_graph(),
      graph.forms(), part, graph.converter());
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

} // namespace tf
