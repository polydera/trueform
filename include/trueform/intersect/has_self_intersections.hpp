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
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/static_size.hpp"
#include "../exact/input_lattice.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../exact/vertex_converter.hpp"
#include "../spatial/aabb_tree.hpp"
#include "../spatial/policy/tree.hpp"
#include "../spatial/tree_config.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/make_face_membership.hpp"
#include "../topology/make_manifold_edge_link.hpp"
#include "../topology/manifold_edge_link.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./face_pairs/find_self_intersection.hpp"
#include "tbb/parallel_invoke.h"

#include <type_traits>

namespace tf {

/// @ingroup intersect
/// @brief Whether the mesh meets itself.
///
/// The verdict of @ref tf::polygon_intersections' one-form build: true
/// exactly when that build would state a self record. This is that build's
/// discovery tier, elections and all, stopped at the first record — so a
/// mesh that meets itself is answered without the rest of it being seen,
/// and a mesh that does not is the whole discovery and nothing above it.
///
/// A shared vertex or edge is not itself a contact; any other touching —
/// between neighbouring faces too — is one, and the lattice decides it
/// exactly.
///
/// Missing structures are built for the call, so a bare mesh is a valid
/// operand; a form already carrying the tree, the face membership and the
/// manifold edge link pays for none of them.
///
/// @tparam Int Exact-integer override (defaulted to @c tf::none_t →
///             resolved from the input coordinate type).
template <typename Int = tf::none_t, typename Policy>
auto has_self_intersections(const tf::polygons<Policy> &polygons) -> bool {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "faces meet each other in three dimensions");
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using RealType = tf::coordinate_type<Policy>;
  constexpr auto Ngon = tf::static_size_v<decltype(polygons.faces()[0])>;

  if (polygons.size() == 0)
    return false;
  if constexpr (!tf::has_tree_policy<Policy> &&
                !tf::has_manifold_edge_link_policy<Policy>) {
    tf::aabb_tree<Index, RealType, 3> tree;
    tf::face_membership<Index> membership;
    tf::manifold_edge_link<Index, Ngon> link;
    tbb::parallel_invoke(
        [&] { tree.build(polygons, tf::config_tree(4, 12)); },
        [&] {
          membership.build(polygons);
          link.build(polygons.faces(), membership);
        });
    return tf::has_self_intersections<Int>(
        polygons | tf::tag(tree) | tf::tag(membership) | tf::tag(link));
  } else if constexpr (!tf::has_tree_policy<Policy>) {
    tf::aabb_tree<Index, RealType, 3> tree;
    tree.build(polygons, tf::config_tree(4, 12));
    return tf::has_self_intersections<Int>(polygons | tf::tag(tree));
  } else if constexpr (!tf::has_face_membership_policy<Policy>) {
    auto membership = tf::make_face_membership(polygons);
    return tf::has_self_intersections<Int>(polygons | tf::tag(membership));
  } else if constexpr (!tf::has_manifold_edge_link_policy<Policy>) {
    auto link = tf::make_manifold_edge_link(polygons);
    return tf::has_self_intersections<Int>(polygons | tf::tag(link));
  } else {
    using resolved_int_type = tf::exact::resolve_int_type<Int, RealType>;
    tf::exact::input_lattice<Index, RealType, resolved_int_type> lattice;
    lattice.build(
        tf::exact::make_vertex_converter<resolved_int_type, RealType>(polygons),
        [&polygons](Index, auto &&f) { f(polygons); }, Index(1), 0.0);
    return tf::intersect::find_self_intersection<Index, resolved_int_type>(
        polygons, lattice);
  }
}

} // namespace tf
