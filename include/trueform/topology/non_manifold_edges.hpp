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
#include "../core/algorithm/generic_generate.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/complete.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/reallocate.hpp"
#include "../core/small_vector.hpp"
#include "../core/views/enumerate.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership.hpp"
#include "./policy/face_membership.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Extract non-manifold edges from faces and face membership.
///
/// Returns all edges that are shared by more than two faces (non-manifold edges).
/// Non-manifold edges indicate problematic mesh topology where more than two
/// faces meet at an edge.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return A @ref tf::blocked_buffer containing pairs of vertex indices for non-manifold edges.
template <typename Policy, typename Policy1>
auto make_non_manifold_edges(const tf::faces<Policy> &faces,
                             const tf::face_membership_like<Policy1> &fm) {
  using Index = std::decay_t<decltype(fm[0][0])>;
  tf::blocked_buffer<Index, 2> edges;
  tf::generic_generate(
      tf::enumerate(faces), edges.data_buffer(), tf::small_vector<Index, 10>{},
      [&](const auto &pair, auto &buffer, auto &neighbors) {
        const auto &[face_id, face] = pair;
        Index size = face.size();
        Index prev = size - 1;
        for (Index i = 0; i < size; prev = i++) {
          neighbors.clear();
          tf::face_edge_neighbors(fm, faces, Index(face_id), Index(face[prev]),
                                  Index(face[i]),
                                  std::back_inserter(neighbors));
          if (neighbors.size() > 1 &&
              // only keep a single copy of an edge
              std::all_of(neighbors.begin(), neighbors.end(),
                          [face_id = Index(face_id)](const auto &x) {
                            return x > face_id;
                          })) {
            buffer.push_back(std::min(face[prev], face[i]));
            buffer.push_back(std::max(face[prev], face[i]));
          }
        }
      });
  return edges;
}

/// @ingroup topology_analysis
/// @brief Extract non-manifold edges from a polygons range.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return A @ref tf::blocked_buffer containing pairs of vertex indices for non-manifold edges.
template <typename Policy>
auto make_non_manifold_edges(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::make_non_manifold_edges(polygons.faces(),
                                       polygons.face_membership());
  } else {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fe;
    fe.build(polygons);
    return tf::make_non_manifold_edges(polygons.faces(), fe);
  }
}

/// @ingroup topology_analysis
/// @brief Extract non-manifold edges with their incident faces.
///
/// Like `tf::make_non_manifold_edges` but additionally returns, for each
/// emitted edge, the block of all face ids incident to that edge
/// (representative face id first, followed by its neighbours across the
/// edge). Edges and face blocks are emitted in matching order.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return A `std::pair` of @ref tf::blocked_buffer "endpoint pairs" and
///         @ref tf::offset_block_buffer "per-edge incident face blocks".
template <typename Policy, typename Policy1>
auto make_non_manifold_edges(const tf::faces<Policy> &faces,
                             const tf::face_membership_like<Policy1> &fm,
                             tf::complete_t) {
  using Index = std::decay_t<decltype(fm[0][0])>;
  std::pair<tf::blocked_buffer<Index, 2>, tf::offset_block_buffer<Index, Index>>
      out;
  struct local_t {
    tf::blocked_buffer<Index, 2> edges;
    tf::buffer<Index> sizes;
    tf::buffer<Index> faces_data;
    tf::small_vector<Index, 10> neighbors;
  };
  Index offset = 0;
  tf::blocked_reduce_sequenced_aggregate(
      tf::enumerate(faces),
      std::tie(out.first, out.second), local_t{},
      [&fm, &faces](const auto &range, local_t &local) {
        for (const auto &pair : range) {
          const auto &[face_id, face] = pair;
          Index size = face.size();
          Index prev = size - 1;
          for (Index i = 0; i < size; prev = i++) {
            local.neighbors.clear();
            tf::face_edge_neighbors(fm, faces, Index(face_id),
                                    Index(face[prev]), Index(face[i]),
                                    std::back_inserter(local.neighbors));
            if (local.neighbors.size() > 1 &&
                std::all_of(local.neighbors.begin(), local.neighbors.end(),
                            [face_id = Index(face_id)](const auto &x) {
                              return x > face_id;
                            })) {
              local.edges.data_buffer().push_back(
                  std::min(face[prev], face[i]));
              local.edges.data_buffer().push_back(
                  std::max(face[prev], face[i]));
              local.sizes.push_back(Index(local.neighbors.size() + 1));
              local.faces_data.push_back(Index(face_id));
              for (auto n : local.neighbors)
                local.faces_data.push_back(n);
            }
          }
        }
      },
      [&offset](const local_t &local,
                std::tuple<tf::blocked_buffer<Index, 2> &,
                           tf::offset_block_buffer<Index, Index> &>
                    result) {
        auto &[edges, faces_blocks] = result;
        if (!local.sizes.size())
          return;
        tf::core::append(local.edges, edges);
        tf::core::append(local.faces_data, faces_blocks.data_buffer());
        for (auto sz : local.sizes) {
          faces_blocks.offsets_buffer().push_back(offset);
          offset += sz;
        }
      });
  if(out.second.offsets_buffer().size())
    out.second.offsets_buffer().push_back(offset);
  return out;
}

/// @ingroup topology_analysis
/// @brief Extract non-manifold edges with incident faces from a polygons range.
/// @overload
template <typename Policy>
auto make_non_manifold_edges(const tf::polygons<Policy> &polygons,
                             tf::complete_t) {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::make_non_manifold_edges(polygons.faces(),
                                       polygons.face_membership(),
                                       tf::complete);
  } else {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fe;
    fe.build(polygons);
    return tf::make_non_manifold_edges(polygons.faces(), fe, tf::complete);
  }
}
} // namespace tf
