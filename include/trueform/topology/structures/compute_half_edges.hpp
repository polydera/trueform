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

#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/circular_decrement.hpp"
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/static_size.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/slide_range.hpp"
#include "../edge_membership.hpp"
#include "../face_edge_neighbors_tagged.hpp"
#include "../face_membership_like.hpp"
#include "../find_eulerian_paths.hpp"
#include "../half_edge.hpp"
#include <array>

namespace tf::topology {

template <typename Index>
auto compute_half_edges_finish(tf::buffer<Index> &id_map,
                               const tf::buffer<Index> &border_ids,
                               tf::buffer<tf::half_edge<Index>> &half_edges,
                               std::size_t n_verts) -> void {
  using half_edge_t = tf::half_edge<Index>;

  // Remap next/prev pointers from flat edge indices to global half-edge indices
  tf::parallel_for_each(half_edges, [&](auto &h) {
    if (h.next != half_edge_t::no_id)
      h.next = id_map[h.next];
    if (h.prev != half_edge_t::no_id)
      h.prev = id_map[h.prev];
  });

  if (border_ids.size() == 0)
    return;

  // Build compact vertex map for border vertices
  id_map.allocate(n_verts);
  tf::parallel_fill(id_map, Index(-1));
  Index current_id = 0;
  for (auto id : border_ids) {
    auto assign = [&](Index he_id) {
      if (id_map[half_edges[he_id].vertex] == Index(-1))
        id_map[half_edges[he_id].vertex] = current_id++;
    };
    assign(id);
    assign(id ^ 1);
  }

  // Build border edges in compact vertex space
  tf::buffer<Index> border_edge_flat;
  border_edge_flat.allocate(border_ids.size() * 2);
  for (Index i = 0; i < Index(border_ids.size()); ++i) {
    auto he_id = border_ids[i];
    border_edge_flat[i * 2] = id_map[half_edges[he_id].vertex];
    border_edge_flat[i * 2 + 1] = id_map[half_edges[he_id ^ 1].vertex];
  }
  auto border_edges =
      tf::make_edges(tf::make_blocked_range<2>(border_edge_flat));

  // Build edge membership for border graph
  tf::edge_membership<Index> em;
  em.build(border_edges, current_id, tf::edge_orientation::forward);

  // Find Eulerian paths
  tf::buffer<Index> path_offsets;
  path_offsets.reserve(border_ids.size());
  tf::buffer<Index> edge_ids;
  edge_ids.reserve(border_ids.size());
  tf::find_eulerian_paths(border_edges, em, path_offsets, edge_ids);

  // Link boundary half-edges along Eulerian paths
  auto border_he_ids = tf::make_indirect_range(edge_ids, border_ids);
  auto paths = tf::make_offset_block_range(path_offsets, border_he_ids);

  for (const auto &path : paths) {
    if (path.size() < 2)
      continue;
    auto last_id = path.back();
    auto first_id = path.front();
    if (half_edges[last_id ^ 1].vertex == half_edges[first_id].vertex) {
      half_edges[last_id].next = first_id;
      half_edges[first_id].prev = last_id;
    }
    for (auto [a, b] : tf::make_slide_range<2>(path)) {
      half_edges[a].next = b;
      half_edges[b].prev = a;
    }
  }
}

/// @brief Compute the half-edge structure from faces and face membership.
///
/// Implements a parallel two-pass algorithm:
/// 1. First pass processes edges where v0 < v1 (by vertex ID), creating
///    half-edge pairs. Each pair consists of the edge and its opposite.
/// 2. Second pass processes remaining edges not covered by the first pass.
/// 3. Finish step remaps next pointers and links boundary half-edges via
///    Eulerian paths.
///
/// Opposite half-edges are stored at adjacent indices (index XOR 1).
///
/// Supports both static-size faces (triangles, quads) and dynamic-size
/// polygon faces.
///
/// @tparam Index The signed integer type for indices.
/// @tparam Faces The faces range type.
/// @tparam Policy The face membership policy type.
/// @param faces The mesh faces.
/// @param fm The face membership structure.
/// @param half_edges Output buffer for half-edge data.
template <typename Index, typename Faces, typename Policy>
auto compute_half_edges(const Faces &faces,
                        const tf::face_membership_like<Policy> &fm,
                        tf::buffer<tf::half_edge<Index>> &half_edges) -> void {
  using half_edge_t = tf::half_edge<Index>;
  constexpr auto StaticN = tf::static_size_v<std::decay_t<decltype(faces[0])>>;

  // Compute edge offset mapping: (face_id, edge_in_face) → flat index
  Index total_edges;
  tf::buffer<Index> edge_offsets_buf;

  if constexpr (StaticN != tf::dynamic_size) {
    total_edges = Index(StaticN) * Index(faces.size());
  } else {
    edge_offsets_buf.allocate(Index(faces.size()) + 1);
    edge_offsets_buf[0] = 0;
    for (Index i = 0; i < Index(faces.size()); ++i)
      edge_offsets_buf[i + 1] = edge_offsets_buf[i] + Index(faces[i].size());
    total_edges = edge_offsets_buf[faces.size()];
  }

  auto edge_offset = [&](Index face_id, Index edge_in_face) -> Index {
    constexpr auto StaticN =
        tf::static_size_v<std::decay_t<decltype(faces[0])>>;
    if constexpr (StaticN != tf::dynamic_size)
      return Index(StaticN) * face_id + edge_in_face;
    else
      return edge_offsets_buf[face_id] + edge_in_face;
  };

  auto get_face_size = [&](Index face_id) -> Index {
    constexpr auto StaticN =
        tf::static_size_v<std::decay_t<decltype(faces[0])>>;
    if constexpr (StaticN != tf::dynamic_size) {
      (void)face_id;
      return Index(StaticN);
    } else {
      return Index(faces[face_id].size());
    }
  };

  // Working structures
  tf::buffer<Index> id_map;
  id_map.allocate(total_edges);

  tf::buffer<std::uint8_t> mask;
  if constexpr (StaticN != tf::dynamic_size && StaticN <= 8) {
    mask.allocate(faces.size());
  } else {
    mask.allocate(total_edges);
  }
  tf::parallel_fill(mask, std::uint8_t(0));

  tf::buffer<Index> border_ids;
  border_ids.reserve(faces.size() / 100 + 1);

  half_edges.reserve(total_edges + total_edges / 10);

  auto log_edge = [&](Index current, Index next_idx, Index face_id,
                      const auto &face, tf::buffer<half_edge_t> &local) {
    std::array<std::array<Index, 2>, 2> nghs;
    auto *ngh_end = tf::face_edge_neighbors_tagged(
        fm, faces, face_id, Index(face[current]), Index(face[next_idx]),
        nghs.data(), nghs.data() + 2);
    auto n_neighbors = ngh_end - nghs.data();

    // next/prev fields temporarily hold the edge-in-face index
    // (remapped to global flat indices in aggregate, then to global
    // half-edge indices in compute_half_edges_finish)
    half_edge_t this_edge{face_id, Index(face[current]), current, current};

    if (n_neighbors == 0) {
      // Boundary edge
      local.push_back(this_edge);
      local.push_back(half_edge_t{half_edge_t::boundary, Index(face[next_idx]),
                                  half_edge_t::no_id, half_edge_t::no_id});
    } else if (n_neighbors == 1) {
      // Simple (manifold) edge
      local.push_back(this_edge);
      const auto &other_face = faces[nghs[0][0]];
      Index other_vertex = other_face[nghs[0][1]];
      if (other_vertex == this_edge.vertex) {
        // Orientation fault: neighbor edge starts at same vertex
        local.push_back(half_edge_t{half_edge_t::orientation_fault,
                                    Index(face[next_idx]), half_edge_t::no_id,
                                    half_edge_t::no_id});
      } else {
        local.push_back(
            half_edge_t{nghs[0][0], other_vertex, nghs[0][1], nghs[0][1]});
      }
    } else {
      // Non-manifold edge
      local.push_back(this_edge);
      local.push_back(half_edge_t{half_edge_t::non_manifold,
                                  Index(face[next_idx]), half_edge_t::no_id,
                                  half_edge_t::no_id});
    }
  };

  auto aggregate = [&](const tf::buffer<half_edge_t> &local,
                       tf::buffer<half_edge_t> &global) {
    constexpr auto StaticN =
        tf::static_size_v<std::decay_t<decltype(faces[0])>>;
    auto current = Index(global.size());
    global.reallocate(current + local.size());
    auto *write_to = global.data() + current;
    for (const auto &h : local) {
      auto modified = h;
      if (modified.face >= 0) {
        if constexpr (StaticN != tf::dynamic_size && StaticN <= 8) {
          Index cur_edge = modified.next;
          Index next_edge = tf::circular_increment(cur_edge, Index(StaticN));
          Index prev_edge = tf::circular_decrement(cur_edge, Index(StaticN));
          mask[modified.face] |= (1 << cur_edge);
          id_map[Index(StaticN) * modified.face + cur_edge] = current;
          modified.next = Index(StaticN) * modified.face + next_edge;
          modified.prev = Index(StaticN) * modified.face + prev_edge;
        } else {
          auto sz = get_face_size(modified.face);
          Index cur_edge = modified.next;
          Index next_edge = tf::circular_increment(cur_edge, sz);
          Index prev_edge = tf::circular_decrement(cur_edge, sz);
          mask[edge_offset(modified.face, cur_edge)] = 1;
          id_map[edge_offset(modified.face, cur_edge)] = current;
          modified.next = edge_offset(modified.face, next_edge);
          modified.prev = edge_offset(modified.face, prev_edge);
        }
      } else if (modified.is_boundary()) {
        border_ids.push_back(current);
      }
      *write_to++ = modified;
      ++current;
    }
  };

  // First pass: process edges where v0 < v1 (by vertex ID)
  tf::blocked_reduce(
      tf::enumerate(faces), half_edges, tf::buffer<half_edge_t>{},
      [&](const auto &r, tf::buffer<half_edge_t> &local) {
        local.reserve(r.size());
        for (const auto &elem : r) {
          auto [fid, face] = elem;
          auto size = face.size();
          decltype(size) prev = size - 1;
          for (decltype(size) i = 0; i < size; prev = i++) {
            if (face[prev] > face[i])
              continue;
            log_edge(Index(prev), Index(i), Index(fid), face, local);
          }
        }
      },
      aggregate);

  using face_type = std::decay_t<decltype(faces[0])>;
  // Second pass: process remaining edges not covered by first pass
  tf::blocked_reduce(
      tf::enumerate(faces), half_edges, tf::buffer<half_edge_t>{},
      [&](const auto &r, tf::buffer<half_edge_t> &local) {
        constexpr auto StaticN = tf::static_size_v<face_type>;
        local.reserve(r.size());
        for (const auto &elem : r) {
          auto [fid, face] = elem;
          if constexpr (StaticN != tf::dynamic_size && StaticN <= 8) {
            constexpr auto filled = (1 << StaticN) - 1;
            auto masked = mask[fid];
            if (masked == filled)
              continue;
            auto size = face.size();
            decltype(size) prev = size - 1;
            for (decltype(size) i = 0; i < size; prev = i++) {
              if (masked & (1 << prev))
                continue;
              log_edge(Index(prev), Index(i), Index(fid), face, local);
            }
          } else {
            auto size = face.size();
            decltype(size) prev = size - 1;
            for (decltype(size) i = 0; i < size; prev = i++) {
              if (mask[edge_offset(Index(fid), Index(prev))])
                continue;
              log_edge(Index(prev), Index(i), Index(fid), face, local);
            }
          }
        }
      },
      aggregate);

  // Finish: remap next pointers and link boundary half-edges
  compute_half_edges_finish(id_map, border_ids, half_edges, fm.size());
}

} // namespace tf::topology
