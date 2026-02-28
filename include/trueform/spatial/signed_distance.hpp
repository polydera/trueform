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
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/distance.hpp"
#include "../core/epsilon.hpp"
#include "../core/policy/normals.hpp"
#include "../core/polygons.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./neighbor_search.hpp"
#include "./policy/tree.hpp"
#include <cmath>

namespace tf {

/// @ingroup spatial_queries
/// @brief Compute the pseudonormal-based signed distance from polygons to a
/// point.
///
/// Positive values indicate the point is outside the surface (in the direction
/// of outward normals), negative values indicate inside.
///
/// The sign is determined using Baerentzen & Aanes' pseudonormal method:
/// - Face interior: sign from face normal
/// - Edge: sign from sum of adjacent face normals
/// - Vertex: sign from angle-weighted sum of incident face normals
///
/// @param polygons The polygon mesh (must have tree, face_membership, and
///                 manifold_edge_link policies).
/// @param point The query point.
/// @return Signed distance (positive = outside, negative = inside).
template <std::size_t Dims, typename Policy0, typename Policy1>
auto signed_distance(const tf::polygons<Policy0> &polygons,
                     const tf::point_like<Dims, Policy1> &point) {
  static_assert(tf::coordinate_dims_v<Policy0> == Dims,
                "Polygon dimensions must match point dimensions");
  static_assert(tf::has_tree_policy<Policy0>, "Use polygons | tf::tag(tree)");
  static_assert(tf::has_face_membership_policy<Policy0>,
                "Use polygons | tf::tag(face_membership)");
  static_assert(tf::has_manifold_edge_link_policy<Policy0>,
                "Use polygons | tf::tag(manifold_edge_link)");

  using RealT = tf::coordinate_type<Policy0, Policy1>;

  auto proj = tf::neighbor_search(polygons, point);
  if (!proj)
    return RealT(0);

  auto frame = tf::frame_of(polygons);
  auto face_id = proj.element;
  auto closest_pt = proj.info.point;
  auto poly = polygons[face_id];
  auto face_normal =
      tf::transformed_normal(tf::make_normal(polygons, face_id), frame);
  auto sz = poly.size();
  auto diff = point - closest_pt;
  constexpr auto eps2 = tf::epsilon2<RealT>;

  // Determine closest feature: vertex, edge, or face interior
  int on_vert = -1;
  {
    RealT best = eps2;
    for (decltype(sz) v = 0; v < sz; ++v) {
      auto d2 = (tf::transformed(poly[v], frame) - closest_pt).length2();
      if (d2 < best) {
        best = d2;
        on_vert = static_cast<int>(v);
      }
    }
  }
  int on_edge = -1;
  if (on_vert < 0) {
    RealT best = eps2;
    auto prev = sz - 1;
    for (decltype(sz) i = 0; i < sz; prev = i++) {
      auto d2 = tf::distance2(
          tf::make_segment_between_points(tf::transformed(poly[prev], frame),
                                          tf::transformed(poly[i], frame)),
          closest_pt);
      if (d2 < best) {
        best = d2;
        on_edge = static_cast<int>(prev);
      }
    }
  }

  // Compute pseudonormal-based sign
  auto dist = tf::sqrt(proj.metric());
  RealT sign_dot;
  if (on_vert >= 0) {
    // Vertex on boundary → outside
    auto vert_id = poly.indices()[on_vert];
    for (auto fi : polygons.face_membership()[vert_id]) {
      for (auto mel : polygons.manifold_edge_link()[fi])
        if (mel.is_boundary())
          return dist;
    }
    // Vertex: angle-weighted normal from all incident faces
    sign_dot = RealT(0);
    for (auto fi : polygons.face_membership()[vert_id]) {
      auto fp = polygons[fi];
      auto fn = tf::transformed_normal(tf::make_normal(polygons, fi), frame);
      auto fsz = fp.size();
      for (decltype(fsz) j = 0; j < fsz; ++j) {
        if (fp.indices()[j] != vert_id)
          continue;
        auto jp = (j + fsz - 1) % fsz;
        auto jn = (j + 1) % fsz;
        auto e1 =
            tf::transformed(fp[jp], frame) - tf::transformed(fp[j], frame);
        auto e2 =
            tf::transformed(fp[jn], frame) - tf::transformed(fp[j], frame);
        auto angle = std::atan2(tf::cross(e1, e2).length(), tf::dot(e1, e2));
        sign_dot += angle * tf::dot(diff, fn);
        break;
      }
    }
  } else if (on_edge >= 0) {
    // Boundary edge → outside
    auto peer = polygons.manifold_edge_link()[face_id][on_edge];
    // Edge: sum of this face normal + peer face normal
    sign_dot = tf::dot(diff, face_normal);
    if (peer.is_simple())
      sign_dot +=
          tf::dot(diff, tf::transformed_normal(
                            tf::make_normal(polygons, peer.face_peer), frame));
  } else {
    // Face interior
    sign_dot = tf::dot(diff, face_normal);
  }

  return sign_dot >= RealT(0) ? dist : -dist;
}

/// @copydoc signed_distance(const tf::polygons<Policy0>&, const
/// tf::point_like<Dims, Policy1>&)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto signed_distance(const tf::point_like<Dims, Policy0> &point,
                     const tf::polygons<Policy1> &polygons) {
  return signed_distance(polygons, point);
}

} // namespace tf
