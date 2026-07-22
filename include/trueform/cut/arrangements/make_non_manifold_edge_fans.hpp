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
#include "../loop_connectivity.hpp"
#include "../face_regions.hpp"
#include "../../core/small_vector.hpp"

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
  /// Per fan entry (parallel to `faces.data_buffer()`): 1 iff that
  /// occurrence traverses the fan's edge as `(edges[i][0] ->
  /// edges[i][1])`. A slit loop appears twice with opposite bits — the
  /// direction is assigned per occurrence at build time (the loop
  /// connection structure defines it), never re-derived from the loop,
  /// where a bridge makes the per-loop question unanswerable.
  tf::buffer<char> dirs;
};

/// @ingroup cut
/// @brief Extract non-manifold edges of an implicit N-form arrangement
///        with their incident REGION fans, from the raw structure.
///
/// Region grain: every region contributes its boundary walk and its
/// hole walks (canonical order — boundary first, then holes in
/// `loop_holes()` order — matching @ref tf::loop_connectivity's edge
/// slots). Only walk edges exist here, so a coincident interior
/// triangulation diagonal can never form a fan — the invariant the
/// loop grain held by construction.
///
/// A non-manifold edge is any walk edge whose cleaned connectivity
/// carries at least one neighbour region; the region and its
/// neighbours form a fan of three or more incident faces.
///
/// Representative rule: only the region with the smallest id in the
/// fan emits. Dead regions (coplanar duplicates) have empty edge
/// ranges in the cleaned connectivity and are skipped.
template <typename Index, typename Int>
auto make_non_manifold_edge_fans(const tf::loop_connectivity<Index> &ag,
                                 const tf::face_regions<Index, Int> &fr)
    -> non_manifold_edge_fans<Index> {
  non_manifold_edge_fans<Index> out;

  auto loops = fr.loops();
  auto holes = fr.holes();
  auto loop_holes = fr.loop_holes();
  if (!loops.size())
    return out;

  auto conn = ag.connectivity_per_face_edge();

  using vertex_t = typename non_manifold_edge_fans<Index>::vertex_t;
  struct local_t {
    tf::blocked_buffer<vertex_t, 2> edges;
    tf::buffer<Index> sizes;
    tf::buffer<Index> faces_data;
    tf::buffer<char> dirs;
  };
  Index offset = 0;
  tf::blocked_reduce(
      tf::enumerate(tf::zip(loops, conn)),
      std::tie(out.edges, out.faces, out.dirs), local_t{},
      [&loops, &holes, &loop_holes](const auto &range, local_t &local) {
        // Walks of a region, canonical order. Applied to a region id.
        auto for_each_walk = [&](Index li, const auto &f) {
          f(loops[li]);
          for (auto h : loop_holes[li])
            f(holes[h]);
        };
        // Undirected / directed traversal counts of {a,b} across ALL
        // of a region's walks (the connectivity self-excludes the
        // region, so its own doubled incidences are invisible there).
        auto edge_mult = [&](Index li, const vertex_t &a,
                             const vertex_t &b) -> Index {
          Index c = 0;
          for_each_walk(li, [&](const auto &w) {
            const Index m = static_cast<Index>(w.size());
            Index q = m - 1;
            for (Index j = 0; j < m; q = j++)
              if ((w[q] == a && w[j] == b) || (w[q] == b && w[j] == a))
                ++c;
          });
          return c;
        };
        auto edge_mult_fwd = [&](Index li, const vertex_t &a,
                                 const vertex_t &b) -> Index {
          Index c = 0;
          for_each_walk(li, [&](const auto &w) {
            const Index m = static_cast<Index>(w.size());
            Index q = m - 1;
            for (Index j = 0; j < m; q = j++)
              if (w[q] == a && w[j] == b)
                ++c;
          });
          return c;
        };
        auto is_forward = [](const vertex_t &a, const vertex_t &b) {
          return a.id != b.id
                     ? a.id < b.id
                     : static_cast<int>(a.source) < static_cast<int>(b.source);
        };
        for (const auto &outer : range) {
          const auto &[loop_id_z, pair] = outer;
          const auto loop_id = static_cast<Index>(loop_id_z);
          const auto &[loop_b, edge_conn_b] = pair;
          const auto &loop = loop_b;
          const auto edge_conn = edge_conn_b;
          if (edge_conn.size() == 0)
            continue; // dead region: cleaned connectivity is empty
          // A doubled edge (slit / pinch) requires a repeated vertex
          // somewhere among the region's walks.
          bool maybe_doubled = false;
          {
            tf::small_vector<vertex_t, 32> seen;
            for_each_walk(loop_id, [&](const auto &w) {
              for (std::size_t i = 0; i < w.size() && !maybe_doubled; ++i) {
                for (const auto &v : seen)
                  if (v == w[i]) {
                    maybe_doubled = true;
                    break;
                  }
                if (!maybe_doubled)
                  seen.push_back(w[i]);
              }
            });
          }
          // Edge slot within a walk follows compute_face_link_per_edge:
          // slot j = (v[j], v[j+1]), closing edge last — i.e. slot ==
          // prev. Walks pack region-major, so a walk's slots start at
          // its running base.
          Index walk_base = 0;
          auto do_walk = [&](const auto &walk) {
            const Index size = static_cast<Index>(walk.size());
            Index prev = size - 1;
            for (Index i = 0; i < size; prev = i++) {
              const auto &neighbours = edge_conn[walk_base + prev];
              const auto &va = walk[prev];
              const auto &vb = walk[i];
              const Index self_mult =
                  maybe_doubled ? edge_mult(loop_id, va, vb) : Index(1);
              const Index total_incidences =
                  self_mult + static_cast<Index>(neighbours.size());
              if (total_incidences < 3)
                continue;
              if (self_mult >= 2 && !is_forward(va, vb))
                continue;
              if (self_mult < 2) {
                bool is_rep = true;
                for (auto n : neighbours)
                  if (n < loop_id) {
                    is_rep = false;
                    break;
                  }
                if (!is_rep)
                  continue;
                bool slit_neighbour = false;
                for (auto n : neighbours)
                  if (edge_mult(n, va, vb) >= 2) {
                    slit_neighbour = true;
                    break;
                  }
                if (slit_neighbour)
                  continue;
              }
              local.edges.data_buffer().push_back(va);
              local.edges.data_buffer().push_back(vb);
              Index fan = 0;
              const Index self_fwd =
                  self_mult >= 2 ? edge_mult_fwd(loop_id, va, vb) : Index(1);
              for (Index k = 0; k < self_mult; ++k) {
                local.faces_data.push_back(loop_id);
                local.dirs.push_back(char(k < self_fwd));
                ++fan;
              }
              for (auto n : neighbours) {
                const Index nm = edge_mult(n, va, vb);
                if (nm == 0) {
                  local.faces_data.push_back(n);
                  local.dirs.push_back(char(1));
                  ++fan;
                  continue;
                }
                const Index nf = edge_mult_fwd(n, va, vb);
                for (Index k = 0; k < nm; ++k) {
                  local.faces_data.push_back(n);
                  local.dirs.push_back(char(k < nf));
                  ++fan;
                }
              }
              local.sizes.push_back(fan);
            }
            walk_base += size;
          };
          do_walk(loop);
          for (auto h : loop_holes[loop_id])
            do_walk(holes[h]);
        }
      },
      [&offset](const local_t &local,
                std::tuple<tf::blocked_buffer<vertex_t, 2> &,
                           tf::offset_block_buffer<Index, Index> &,
                           tf::buffer<char> &>
                    result) {
        auto &[edges, faces_blocks, dirs] = result;
        if (!local.sizes.size())
          return;
        tf::core::append(local.edges, edges);
        tf::core::append(local.faces_data, faces_blocks.data_buffer());
        tf::core::append(local.dirs, dirs);
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
