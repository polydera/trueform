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
#include "../reindex/return_index_map.hpp"
#include "../reindex/return_source_ids.hpp"
#include "../topology/domain_config.hpp"
#include "./csg_graph.hpp"
#include "./graph/open_fragments_for.hpp"
#include "./make_csg_domains.hpp"

namespace tf {

/// @ingroup csg
/// @brief Per form, the live triangles of each component whose two coarse
///        sides under `config` are one domain, wound as stored: what the
///        cells of @ref tf::make_csg_domains leave out under the same config.
/// @return `{ fragments, tags }`: one mesh per form, `tags[k] == k`.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_open_fragments(
    const tf::csg_graph<Policy, Int, Arrangement> &graph,
    tf::domain_config config) {
  return csg::graph::open_fragments_for<OutputCoordinateType>(graph, config);
}

/// @ingroup csg
/// @brief Default flag set, that of @ref tf::make_csg_domains.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_open_fragments(
    const tf::csg_graph<Policy, Int, Arrangement> &graph) {
  return make_csg_open_fragments<OutputCoordinateType>(
      graph, csg::csg_domains_default_config);
}

/// @ingroup csg
/// @brief Explicit `config`, returning per-fragment (tag, face) provenance:
///        `(fragments, tags, tag_blocks, face_blocks)`.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_open_fragments(
    const tf::csg_graph<Policy, Int, Arrangement> &graph,
    tf::domain_config config, tf::return_source_ids_t) {
  return csg::graph::open_fragments_for<OutputCoordinateType, true>(graph,
                                                                    config);
}

/// @ingroup csg
/// @brief Default flags, returning per-fragment provenance.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_open_fragments(
    const tf::csg_graph<Policy, Int, Arrangement> &graph,
    tf::return_source_ids_t) {
  return make_csg_open_fragments<OutputCoordinateType>(
      graph, csg::csg_domains_default_config, tf::return_source_ids);
}

/// @ingroup csg
/// @brief Explicit `config`, returning a @ref tf::csg_domains_index_map:
///        `(fragments, tags, index_map)`.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_open_fragments(
    const tf::csg_graph<Policy, Int, Arrangement> &graph,
    tf::domain_config config, tf::return_index_map_t) {
  return csg::graph::open_fragments_for<OutputCoordinateType, true, true>(
      graph, config);
}

/// @ingroup csg
/// @brief Default flags, returning a @ref tf::csg_domains_index_map.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Int, template <typename, typename> class Arrangement>
auto make_csg_open_fragments(
    const tf::csg_graph<Policy, Int, Arrangement> &graph,
    tf::return_index_map_t) {
  return make_csg_open_fragments<OutputCoordinateType>(
      graph, csg::csg_domains_default_config, tf::return_index_map);
}

} // namespace tf
