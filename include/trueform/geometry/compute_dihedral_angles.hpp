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
#include "../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../core/angle.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/policy/normals.hpp"
#include "../core/polygons.hpp"
#include "../core/reallocate.hpp"
#include "../core/views/enumerate.hpp"
#include "../topology/make_manifold_edge_link.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./compute_normals.hpp"
#include "./quality/dihedral_angle.hpp"
#include <algorithm>
#include <type_traits>

namespace tf {

/// @ingroup geometry
/// @brief The dihedral angles of a mesh: `edges` blocked in vertex pairs and
/// `angles` aligned with it, face-major.
///
/// One entry per undirected edge, reached through the manifold edge link's
/// representative. A boundary or non-manifold edge joins no pair of faces and
/// turns through no angle, so it is not stated.
template <typename Index, typename T> struct dihedral_angles {
  tf::blocked_buffer<Index, 2> edges;
  tf::buffer<tf::rad<T>> angles;
};

/// @ingroup geometry
/// @brief Measure every edge two faces of a mesh share.
///
/// Uses a tagged manifold edge link and tagged normals when the form carries
/// them, and builds whichever is missing for the call.
///
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input polygons (must be 3D).
/// @return A @ref tf::dihedral_angles holding the edges and their angles.
template <typename Policy>
auto compute_dihedral_angles(const tf::polygons<Policy> &polygons) {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "compute_dihedral_angles requires 3D polygons");
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using T = tf::coordinate_type<Policy>;

  struct local_t {
    tf::buffer<Index> edges;
    tf::buffer<tf::rad<T>> angles;
  };

  auto run = [&polygons](const auto &mel, const auto &normals) {
    tf::dihedral_angles<Index, T> result;
    tf::blocked_reduce_sequenced_aggregate(
        tf::enumerate(polygons.faces()), result, local_t{},
        [&mel, &normals](const auto &block, local_t &local) {
          local.edges.clear();
          local.angles.clear();
          for (const auto &pair : block) {
            const auto &[face_id, face] = pair;
            Index size = face.size();
            Index prev = size - 1;
            for (Index i = 0; i < size; prev = i++) {
              auto &&peer = mel[face_id][prev];
              if (!peer.is_simple())
                continue;
              if (!peer.is_representative(Index(face_id)))
                continue;
              local.edges.push_back(std::min(face[prev], face[i]));
              local.edges.push_back(std::max(face[prev], face[i]));
              local.angles.push_back(tf::geometry::dihedral_angle(
                  normals[face_id], normals[peer.face_peer]));
            }
          }
        },
        [](const local_t &local, tf::dihedral_angles<Index, T> &out) {
          tf::core::append(local.edges, out.edges.data_buffer());
          tf::core::append(local.angles, out.angles);
        },
        tf::checked);
    return result;
  };

  auto dispatch_normals = [&](const auto &mel) {
    if constexpr (tf::has_normals_policy<Policy>) {
      return run(mel, polygons.normals());
    } else {
      auto normals = tf::compute_normals(polygons);
      return run(mel, normals.unit_vectors());
    }
  };

  if constexpr (tf::has_manifold_edge_link_policy<Policy>) {
    return dispatch_normals(polygons.manifold_edge_link());
  } else {
    auto mel = tf::make_manifold_edge_link(polygons);
    return dispatch_normals(mel);
  }
}

} // namespace tf
