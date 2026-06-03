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
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../arrangement_graph.hpp"
#include "../face_cuts.hpp"

namespace tf::cut {

/// @ingroup cut
/// @brief Owning container for non-manifold edge fans of an implicit
///        N-form arrangement.
///
/// Two parallel containers indexed by non-manifold-edge id:
/// - `edges[i]`: the two `intersect::graph::vertex` endpoints of the
///   edge, taken directly from the representative loop at the relevant
///   edge slot.
/// - `faces[i]`: the incident loop fan — representative loop first,
///   then the neighbour loops across the edge.
///
/// Build via @ref tf::cut::make_non_manifold_edge_fans.
template <typename Index> struct non_manifold_edge_fans {
  using vertex_t = tf::intersect::graph::vertex<Index>;
  tf::blocked_buffer<vertex_t, 2> edges;
  tf::offset_block_buffer<Index, Index> faces;
};

/// @ingroup cut
/// @brief Extract non-manifold edges of an implicit N-form arrangement
///        with their incident loop fans.
///
/// A non-manifold edge of the arrangement is any per-loop edge whose
/// cleaned connectivity carries at least one neighbour: the loop and
/// its neighbours form a fan of three or more incident faces across
/// that edge.
///
/// Representative rule: only the loop with the smallest id in the fan
/// emits — mirrors the standard pattern from
/// @ref tf::make_non_manifold_edge_fans on materialised polygons.
///
/// Dead loops (coplanar duplicates dropped during
/// @ref tf::arrangement_graph::build) have empty per-loop edge ranges
/// in the graph's connectivity and are skipped here.
template <typename Index, typename Int>
auto make_non_manifold_edge_fans(const tf::arrangement_graph<Index> &ag,
                                 const tf::face_cuts<Index, Int> &fc)
    -> non_manifold_edge_fans<Index> {
  non_manifold_edge_fans<Index> out;

  auto loops = fc.loops();
  if (!loops.size())
    return out;

  auto conn = ag.connectivity_per_face_edge();

  using vertex_t = typename non_manifold_edge_fans<Index>::vertex_t;
  struct local_t {
    tf::blocked_buffer<vertex_t, 2> edges;
    tf::buffer<Index> sizes;
    tf::buffer<Index> faces_data;
  };
  Index offset = 0;
  tf::blocked_reduce(
      tf::enumerate(tf::zip(loops, conn)), std::tie(out.edges, out.faces),
      local_t{},
      [](const auto &range, local_t &local) {
        for (const auto &outer : range) {
          const auto &[loop_id_z, pair] = outer;
          const auto loop_id = static_cast<Index>(loop_id_z);
          const auto &[loop, edge_conn] = pair;
          // Dead loops (coplanar duplicates) keep their raw vertex
          // count in fc.loops(), but their cleaned connectivity is
          // empty. Skip them; bypassing this check would read
          // edge_conn[k] past its tail and pull garbage neighbours.
          if (edge_conn.size() == 0)
            continue;
          const Index size = static_cast<Index>(loop.size());
          Index prev = size - 1;
          for (Index i = 0; i < size; prev = i++) {
            const auto &neighbours = edge_conn[prev];
            // NM means K > 2 incident faces — i.e. the loop has > 1
            // peers at this edge. Size 1 is a normal manifold edge.
            if (neighbours.size() < 2)
              continue;
            bool is_rep = true;
            for (auto n : neighbours) {
              if (n < loop_id) {
                is_rep = false;
                break;
              }
            }
            if (!is_rep)
              continue;
            local.edges.data_buffer().push_back(loop[prev]);
            local.edges.data_buffer().push_back(loop[i]);
            local.sizes.push_back(Index(neighbours.size() + 1));
            local.faces_data.push_back(loop_id);
            for (auto n : neighbours)
              local.faces_data.push_back(n);
          }
        }
      },
      [&offset](const local_t &local,
                std::tuple<tf::blocked_buffer<vertex_t, 2> &,
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

} // namespace tf::cut
