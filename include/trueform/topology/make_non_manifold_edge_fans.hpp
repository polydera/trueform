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
#include "../core/algorithm/block_reduce.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/reallocate.hpp"
#include "../core/small_vector.hpp"
#include "../core/views/enumerate.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./non_manifold_edge_fans.hpp"
#include "./policy/face_membership.hpp"
#include "./policy/manifold_edge_link.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Extract non-manifold edges with their incident face fans.
///
/// At each non-manifold edge, three or more faces meet. The returned
/// @ref tf::non_manifold_edge_fans carries two parallel containers:
/// per-edge endpoint pairs (`edges`) and per-edge incident face blocks
/// (`faces`, representative face first, then neighbours across the
/// edge).
template <typename Policy, typename Policy1>
auto make_non_manifold_edge_fans(const tf::faces<Policy> &faces,
                                 const tf::face_membership_like<Policy1> &fm) {
  using Index = std::decay_t<decltype(fm[0][0])>;
  tf::non_manifold_edge_fans<Index> out;
  struct local_t {
    tf::blocked_buffer<Index, 2> edges;
    tf::buffer<Index> sizes;
    tf::buffer<Index> faces_data;
    tf::small_vector<Index, 10> neighbors;
  };
  Index offset = 0;
  tf::blocked_reduce(
      tf::enumerate(faces), std::tie(out.edges, out.faces), local_t{},
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
  if (out.faces.offsets_buffer().size())
    out.faces.offsets_buffer().push_back(offset);
  return out;
}

/// @ingroup topology_analysis
/// @brief Extract non-manifold edge fans, using manifold-edge link to
///        prune the per-face edge scan to the non-manifold edges only.
/// @overload
template <typename Policy, typename Policy1, typename Policy2>
auto make_non_manifold_edge_fans(
    const tf::faces<Policy> &faces, const tf::face_membership_like<Policy1> &fm,
    const tf::manifold_edge_link_like<Policy2> &mel) {
  using Index = std::decay_t<decltype(fm[0][0])>;
  tf::non_manifold_edge_fans<Index> out;
  struct local_t {
    tf::blocked_buffer<Index, 2> edges;
    tf::buffer<Index> sizes;
    tf::buffer<Index> faces_data;
    tf::small_vector<Index, 10> neighbors;
  };
  Index offset = 0;
  tf::blocked_reduce(
      tf::enumerate(tf::zip(faces, mel)), std::tie(out.edges, out.faces),
      local_t{},
      [&fm, &faces](const auto &range, local_t &local) {
        for (const auto &_pair : range) {
          const auto &[face_id, pair] = _pair;
          const auto &[face, mel] = pair;
          Index size = face.size();
          Index prev = size - 1;
          for (Index i = 0; i < size; prev = i++) {
            if (mel[prev].is_manifold())
              continue;
            if (!mel[prev].is_representative(Index(face_id)))
              continue;
            local.neighbors.clear();
            tf::face_edge_neighbors(fm, faces, Index(face_id),
                                    Index(face[prev]), Index(face[i]),
                                    std::back_inserter(local.neighbors));
            if (local.neighbors.size() > 1) {
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
  if (out.faces.offsets_buffer().size())
    out.faces.offsets_buffer().push_back(offset);
  return out;
}

/// @ingroup topology_analysis
/// @brief Extract non-manifold edge fans from a polygons range.
///
/// Reuses any tagged @ref tf::face_membership and
/// @ref tf::manifold_edge_link from the policy; builds what's missing.
/// @overload
template <typename Policy>
auto make_non_manifold_edge_fans(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::has_face_membership_policy<Policy> &&
                !tf::has_manifold_edge_link_policy<Policy>) {
    return tf::make_non_manifold_edge_fans(polygons.faces(),
                                           polygons.face_membership());
  } else if constexpr (tf::has_face_membership_policy<Policy> &&
                       tf::has_manifold_edge_link_policy<Policy>) {
    return tf::make_non_manifold_edge_fans(polygons.faces(),
                                           polygons.face_membership(),
                                           polygons.manifold_edge_link());
  } else if constexpr (!tf::has_face_membership_policy<Policy> &&
                       tf::has_manifold_edge_link_policy<Policy>) {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fe;
    fe.build(polygons);
    return tf::make_non_manifold_edge_fans(polygons.faces(), fe,
                                           polygons.manifold_edge_link());
  } else {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fe;
    fe.build(polygons);
    return tf::make_non_manifold_edge_fans(polygons.faces(), fe);
  }
}

} // namespace tf
