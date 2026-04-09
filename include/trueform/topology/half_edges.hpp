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

#include <tuple>

#include "tbb/parallel_invoke.h"

#include "../core/algorithm/mask_to_index_map.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/index_map.hpp"
#include "../core/polygons.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "./half_edges_like.hpp"
#include "./make_face_membership.hpp"
#include "./policy/face_membership.hpp"
#include "./structures/compute_half_edges.hpp"
#include "./structures/half_edges_buffers.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Half-edge mesh data structure.
///
/// Stores half-edges in a flat buffer where opposite half-edges are at
/// adjacent indices (index XOR 1). Provides navigation (next, previous,
/// opposite, rotated), queries (boundary, manifold), and operations (flip).
///
/// All navigation and query methods are safe by default — they propagate
/// invalid handles. Pass `tf::unsafe` as the first argument to skip
/// validity checks for performance-critical inner loops.
///
/// Build from polygons via the constructor or `build()` method. Supports
/// both static-size faces (triangles, quads) and dynamic-size polygon faces.
///
/// @tparam Index The signed integer type for indices.
template <typename Index>
class half_edges : public half_edges_like<topology::half_edges_buffers<Index>> {
  using base_t = half_edges_like<topology::half_edges_buffers<Index>>;

public:
  using typename base_t::edge_handle_t;
  using typename base_t::face_handle_t;
  using typename base_t::half_edge_handle_t;
  using typename base_t::index_type;
  using typename base_t::vertex_handle_t;

  half_edges() = default;

  template <typename Policy>
  explicit half_edges(const tf::polygons<Policy> &polygons) {
    build(polygons);
  }

  template <typename Policy>
  auto build(const tf::polygons<Policy> &polygons) -> void {
    if constexpr (tf::has_face_membership_policy<Policy>) {
      build(polygons.faces(), polygons.face_membership());
    } else {
      auto fm = tf::make_face_membership(polygons);
      build(polygons | tf::tag(fm));
    }
  }

  template <typename Faces, typename Policy>
  auto build(const Faces &faces, const tf::face_membership_like<Policy> &fm)
      -> void {
    this->_half_edges.clear();
    tf::topology::compute_half_edges<Index>(faces, fm, this->_half_edges);
    build_handle_buffers(Index(faces.size()), Index(fm.size()));
  }

  auto clear() -> void {
    this->_half_edges.clear();
    this->_face_half_edges.clear();
    this->_vertex_half_edges.clear();
    this->_boundary_vertices.clear();
    this->_non_manifold_vertices.clear();
    this->_n_faces = 0;
    this->_n_vertices = 0;
  }

  auto rebuild_handles(Index n_faces, Index n_verts) -> void {
    build_handle_buffers(n_faces, n_verts);
  }

  auto compact()
      -> std::tuple<tf::index_map_buffer<Index>, tf::index_map_buffer<Index>,
                    tf::index_map_buffer<Index>> {
    Index n_edges = Index(this->_half_edges.size() / 2);
    Index n_old_faces = Index(this->_face_half_edges.size());
    Index n_old_verts = Index(this->_vertex_half_edges.size());

    tf::index_map_buffer<Index> edge_im;
    tf::index_map_buffer<Index> face_im;
    tf::index_map_buffer<Index> vert_im;

    tbb::parallel_invoke(
        [&] {
          tf::mask_to_index_map(
              tf::make_mapped_range(
                  tf::make_sequence_range(n_edges),
                  [&](Index e) {
                    return !this->_half_edges[e << 1].is_removed();
                  }),
              edge_im);
        },
        [&] {
          tf::mask_to_index_map(
              tf::make_mapped_range(
                  tf::make_sequence_range(n_old_faces),
                  [&](Index f) {
                    return this->_face_half_edges[f].is_valid();
                  }),
              face_im);
        },
        [&] {
          tf::mask_to_index_map(
              tf::make_mapped_range(
                  tf::make_sequence_range(n_old_verts),
                  [&](Index v) {
                    return this->_vertex_half_edges[v].is_valid();
                  }),
              vert_im);
        });

    Index n_new_edges = Index(edge_im.kept_ids().size());
    tf::buffer<tf::half_edge<Index>> new_hes;
    new_hes.allocate(n_new_edges * 2);
    auto &em = edge_im.f();
    auto &fm = face_im.f();
    auto &vm = vert_im.f();
    tf::parallel_for_each(tf::make_sequence_range(n_edges), [&](Index e) {
      if (em[e] == Index(em.size()))
        return;
      for (int side = 0; side < 2; ++side) {
        auto &h = this->_half_edges[(e << 1) + side];
        auto &nh = new_hes[(em[e] << 1) + side];
        nh.face = h.face >= 0 ? fm[h.face] : h.face;
        nh.vertex = vm[h.vertex];
        nh.next = h.has_next() ? (em[h.next >> 1] << 1) + (h.next & 1) : h.next;
        nh.prev = h.prev >= 0 ? (em[h.prev >> 1] << 1) + (h.prev & 1) : h.prev;
      }
    });
    this->_half_edges = std::move(new_hes);

    Index n_new_faces = Index(face_im.kept_ids().size());
    Index n_new_verts = Index(vert_im.kept_ids().size());

    build_handle_buffers(n_new_faces, n_new_verts);

    return {std::move(face_im), std::move(vert_im), std::move(edge_im)};
  }

private:
  auto build_handle_buffers(Index n_faces, Index n_verts) -> void {
    this->_n_faces = n_faces;
    this->_n_vertices = n_verts;
    this->_face_half_edges.allocate(n_faces);
    tf::parallel_fill(this->_face_half_edges, half_edge_handle_t::invalid());
    this->_vertex_half_edges.allocate(n_verts);
    tf::parallel_fill(this->_vertex_half_edges, half_edge_handle_t::invalid());
    this->_boundary_vertices.allocate(n_verts);
    tf::parallel_fill(this->_boundary_vertices, char(0));
    this->_non_manifold_vertices.allocate(n_verts);
    tf::parallel_fill(this->_non_manifold_vertices, char(0));
    tf::parallel_for_each(
        tf::make_mapped_range(
            tf::make_sequence_range(Index(this->_half_edges.size())),
            [](Index id) { return half_edge_handle_t{id}; }),
        [&](const auto &heh) {
          const auto &h = this->_half_edges[heh.id()];
          if (h.is_simple()) {
            this->_face_half_edges[h.face] = heh;
            this->_vertex_half_edges[h.vertex] = heh;
          } else if (h.is_boundary()) {
            this->_boundary_vertices[h.vertex] = 1;
          }
        });

    // Detect non-manifold vertices: compare sequential per-vertex incident
    // count to the number of half-edges reachable by walking rotated() from
    // the vertex's representative. A mismatch means the vertex has a split
    // fan or is incident to a non-manifold edge sentinel that rotated() can
    // not cross — either case produces stale .vertex pointers during
    // collapse and we must forbid it in is_collapse_ok.
    tf::buffer<Index> vertex_degree;
    vertex_degree.allocate(n_verts);
    tf::parallel_fill(vertex_degree, Index(0));
    for (const auto &h : this->_half_edges) {
      if (h.vertex >= 0 && h.vertex < n_verts)
        ++vertex_degree[h.vertex];
    }
    tf::parallel_for_each(
        tf::make_sequence_range(n_verts),
        [&](Index v) {
          auto seed = this->_vertex_half_edges[v];
          if (!seed.is_valid()) {
            this->_non_manifold_vertices[v] = 1;
            return;
          }
          Index reachable = 0;
          auto cur = seed;
          do {
            ++reachable;
            auto nxt = this->rotated(cur);
            if (!nxt.is_valid())
              break;
            cur = nxt;
          } while (cur != seed);
          if (reachable < vertex_degree[v])
            this->_non_manifold_vertices[v] = 1;
        });
  }
};

} // namespace tf
