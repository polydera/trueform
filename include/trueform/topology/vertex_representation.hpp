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

#include "../core/faces.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/mapped_range.hpp"
#include "./face_membership_like.hpp"

namespace tf {
namespace topology {

template <typename Policy> struct vertex_representation_inner_dref {

  template <typename T> auto operator()(T &&v_id) const {
    return std::size_t(_fe[v_id].front()) == face_id;
  }
  std::size_t face_id;
  tf::face_membership_like<Policy> _fe;
};

template <typename Policy> struct vertex_representation_dref {
  tf::face_membership_like<Policy> _fe;
  //
  template <typename T> auto operator()(T &&pair) const {
    auto &&[face_id, face] = pair;
    return tf::make_mapped_range(face, vertex_representation_inner_dref<Policy>{
                                           std::size_t(face_id), _fe});
  }
};
} // namespace topology

/// @ingroup topology_types
/// @brief Creates a range indicating which vertices in a face are representatives.
///
/// For each vertex in the specified face, returns true if this face "owns"
/// the vertex (is the first face in the vertex's face membership list),
/// false otherwise. This is useful for avoiding duplicate processing of
/// shared vertices.
///
/// @tparam Range The face range type.
/// @tparam Policy The face membership policy type.
/// @param face_id The face to query.
/// @param face The vertex indices of the face.
/// @param fe The face membership structure.
/// @return A range of booleans indicating representative status per vertex.
template <typename Range, typename Policy>
auto make_vertex_representation(std::size_t face_id, const Range &face,
                                const tf::face_membership_like<Policy> &fe) {
  auto r = tf::make_range(fe);
  return tf::make_mapped_range(
      face, topology::vertex_representation_inner_dref<decltype(r)>{
                face_id, tf::make_face_membership_like(std::move(r))});
}

/// @ingroup topology_types
/// @brief Creates a nested range of vertex representation flags for all faces.
///
/// For each face and each vertex within it, indicates whether this face
/// is the representative owner of that vertex.
///
/// @tparam Policy0 The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The mesh faces.
/// @param fe The face membership structure.
/// @return A nested range of booleans per face, per vertex.
template <typename Policy0, typename Policy1>
auto make_vertex_representation(const tf::faces<Policy0> &faces,
                                const tf::face_membership_like<Policy1> &fe) {
  auto r = tf::make_range(fe);
  return tf::make_mapped_range(
      tf::enumerate(faces), topology::vertex_representation_dref<decltype(r)>{
                                tf::make_face_membership_like(std::move(r))});
}
} // namespace tf
