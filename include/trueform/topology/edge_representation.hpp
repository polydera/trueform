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

#include "../core/views/enumerate.hpp"
#include "../core/views/mapped_range.hpp"
#include "./manifold_edge_link_like.hpp"

namespace tf {
namespace topology {

template <typename Index> struct edge_representation_inner_dref {

  template <typename T> auto operator()(T &&fp) const {
    return fp.is_representative(face_id);
  }
  Index face_id;
};

template <typename Index> struct edge_representation_dref {
  template <typename T> auto operator()(T &&pair) const {
    auto &&[id, link] = pair;
    return tf::make_mapped_range(
        link, edge_representation_inner_dref<Index>{Index(id)});
  }
};
} // namespace topology

/// @ingroup topology_types
/// @brief Creates a range indicating which edges in a face are representatives.
///
/// For each edge in the specified face, returns true if this face "owns"
/// the edge (is the representative), false otherwise. This is useful for
/// avoiding duplicate processing of shared edges.
///
/// @tparam Index The integer type for indices.
/// @tparam Policy The manifold edge link policy type.
/// @param face_id The face to query.
/// @param mel The manifold edge link structure.
/// @return A range of booleans indicating representative status per edge.
template <typename Index, typename Policy>
auto make_edge_representation(Index face_id,
                              const tf::manifold_edge_link_like<Policy> &mel) {
  return tf::make_mapped_range(
      mel[face_id],
      topology::edge_representation_inner_dref<Index>{Index(face_id)});
}

/// @ingroup topology_types
/// @brief Creates a nested range of edge representation flags for all faces.
///
/// For each face and each edge within it, indicates whether this face
/// is the representative owner of that edge.
///
/// @tparam Policy The manifold edge link policy type.
/// @param mel The manifold edge link structure.
/// @return A nested range of booleans per face, per edge.
template <typename Policy>
auto make_edge_representation(const tf::manifold_edge_link_like<Policy> &mel) {
  using Index = std::decay_t<decltype(mel[0][0].face_peer)>;
  return tf::make_mapped_range(tf::enumerate(mel),
                               topology::edge_representation_dref<Index>{});
}
} // namespace tf
