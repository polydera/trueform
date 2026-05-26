/*
 * Copyright (c) 2025 XLAB
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
#include "../core/buffer.hpp"
#include "../core/none.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/point.hpp"
#include "../core/polygons.hpp"
#include "../core/tagged_sidedness.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../exact_coordinate_converter.hpp"
#include "./make_face_membership.hpp"
#include "./make_manifold_edge_connected_component_labels.hpp"
#include "./make_manifold_edge_link.hpp"
#include "./make_non_manifold_edge_fans.hpp"
#include "./policy/connected_component_labels.hpp"
#include "./policy/face_membership.hpp"
#include "./policy/manifold_edge_link.hpp"
#include "./sidedness/aggregate_votes.hpp"
#include "./sidedness/emit_votes.hpp"
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup topology_components
/// @brief Per-component sidedness against each operand mesh at the cuts.
///
/// For each manifold-edge-connected component of an arrangement, emits
/// a block of @ref tf::tagged_sidedness entries listing the operand
/// tags the component shares an intersection edge with and the
/// sidedness of the component against that operand's oriented surface.
///
/// The arrangement is consumed read-only; required topology structures
/// (face_membership, manifold_edge_link) are built and tagged
/// recursively if absent. Connected component labels are either reused
/// from the tagged form or built internally.
///
/// @tparam Int Exact integer type for predicates. Default `tf::none_t`
///         resolves via @ref tf::exact::resolve_int_type.
/// @tparam Policy The polygons policy type.
/// @tparam Range The tag-labels container type.
/// @param polygons The arrangement.
/// @param tag_labels Per-face source-mesh tag (from `mesh_arrangements`).
/// @return When the form has `connected_component_labels` tagged:
///         the relations block alone. Otherwise a
///         `std::pair{relations, owning connected_component_labels}`.
template <typename Int = tf::none_t, typename Policy, typename Range>
auto make_sidedness_relations(const tf::polygons<Policy> &polygons,
                              const Range &tag_labels) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using CoordType = tf::coordinate_type<Policy>;
  using ResolvedInt = tf::exact::resolve_int_type<Int, CoordType>;

  if constexpr (!tf::has_face_membership_policy<Policy>) {
    auto fm = tf::make_face_membership(polygons);
    return make_sidedness_relations<Int>(polygons | tf::tag(fm), tag_labels);
  } else if constexpr (!tf::has_manifold_edge_link_policy<Policy>) {
    auto mel = tf::make_manifold_edge_link(polygons);
    return make_sidedness_relations<Int>(polygons | tf::tag(mel), tag_labels);
  } else {
    auto component_labels = [&]() {
      if constexpr (tf::has_connected_component_labels_policy<Policy>) {
        return polygons.connected_component_labels();
      } else {
        return tf::make_manifold_edge_connected_component_labels(polygons);
      }
    }();

    auto [nm_edges, nm_edge_faces] = tf::make_non_manifold_edge_fans(polygons);

    auto get_point = [&]() {
      if constexpr (std::is_integral_v<CoordType>) {
        return [&polygons](Index v) -> tf::point<ResolvedInt, 3> {
          auto p = polygons.points()[v];
          return tf::point<ResolvedInt, 3>{ResolvedInt(p[0]), ResolvedInt(p[1]),
                                           ResolvedInt(p[2])};
        };
      } else {
        auto conv = tf::make_exact_coordinate_converter<ResolvedInt>(polygons);
        return [conv, &polygons](Index v) -> tf::point<ResolvedInt, 3> {
          return conv(polygons.points()[v]);
        };
      }
    }();

    tf::buffer<Index> vote_components;
    tf::buffer<tf::tagged_sidedness<Index>> vote_entries;
    tf::topology::sidedness::emit_votes(polygons, tag_labels, component_labels,
                                        nm_edges, nm_edge_faces, get_point,
                                        vote_components, vote_entries);

    tf::offset_block_buffer<Index, tf::tagged_sidedness<Index>> relations;
    tf::topology::sidedness::aggregate_votes(
        vote_components, vote_entries,
        static_cast<Index>(component_labels.n_components), relations);

    if constexpr (tf::has_connected_component_labels_policy<Policy>) {
      return relations;
    } else {
      return std::make_pair(std::move(relations), std::move(component_labels));
    }
  }
}

} // namespace tf
