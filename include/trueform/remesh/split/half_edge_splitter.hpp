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

#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/slice.hpp"
#include "../../core/views/take.hpp"
#include "../../topology/half_edges.hpp"

#include <array>

namespace tf {
namespace remesh {

template <typename Index, typename Real, std::size_t Dims>
class half_edge_splitter {
  using half_edge_handle_t = tf::half_edge_handle<Index>;
  using edge_handle_t = tf::edge_handle<Index>;
  using vertex_handle_t = tf::vertex_handle<Index>;
  using face_handle_t = tf::face_handle<Index>;
  using he_t = tf::half_edge<Index>;

public:
  template <typename PointsPolicy, typename Handler>
  auto split(const tf::points<PointsPolicy> &point_data,
             tf::half_edges<Index> &he, const Handler &handler,
             bool preserve_boundary) -> Index {
    clear();
    Index n_original_half_edges = Index(he.half_edges_buffer().size());
    compute_edge_into_split_point_map(point_data, he, handler,
                                      preserve_boundary);
    _previous_boundary_edge.allocate(he.half_edges_buffer().size() / 2);
    Index n_polys = build_half_edges_from_map(
        he, he.face_half_edge_handles(), Index(point_data.size()));
    write_computed_halfedges(he);
    if (!preserve_boundary)
      relink_boundary_edges(he, n_original_half_edges);
    return n_polys;
  }

  auto created_point_data() const -> const tf::points_buffer<Real, Dims> & {
    return _point_data;
  }

  auto clear() -> void {
    _outer_half_edges.clear();
    _inner_half_edges.clear();
    _edge_to_point_map.clear();
    _point_data.clear();
    _previous_boundary_edge.clear();
  }

private:
  // --- Phase 1: find edges to split, create midpoints ---

  template <typename PointsPolicy, typename Handler>
  auto compute_edge_into_split_point_map(const tf::points<PointsPolicy> &point_data,
                                         tf::half_edges<Index> &he,
                                         const Handler &handler,
                                         bool preserve_boundary) -> void {
    Index n_edges = Index(he.half_edges_buffer().size() / 2);
    _edge_to_point_map.allocate(n_edges);
    tf::parallel_fill(_edge_to_point_map, _clean_tag);

    struct local_data {
      tf::points_buffer<Real, Dims> pts;
      tf::buffer<Index> ids;
    };
    struct aggregate_data {
      tf::points_buffer<Real, Dims> &point_data;
      tf::buffer<Index> &edge_to_point_map;
      Index all_pts;
    };

    aggregate_data agg{_point_data, _edge_to_point_map,
                       Index(point_data.size())};

    tf::blocked_reduce(
        he.edge_handles(), agg, local_data{},
        [&he, &point_data, &handler, preserve_boundary](auto r,
                                                         local_data &local) {
          for (auto eh : r) {
            if (!eh.is_valid())
              continue;
            if (!he.is_manifold(tf::unsafe, eh))
              continue;
            if (preserve_boundary && he.is_boundary(tf::unsafe, eh))
              continue;
            if (handler.should_skip(eh))
              continue;
            auto heh0 = he.half_edge_handle(tf::unsafe, eh, false);
            auto d2 = handler.distance2(he, point_data, heh0);
            if (d2 > handler.max_length2(eh)) {
              local.pts.push_back(handler.interpolate(he, point_data, heh0));
              local.ids.push_back(eh.id());
            }
          }
        },
        [](const local_data &local, aggregate_data &agg) {
          for (std::size_t i = 0; i < local.pts.size(); ++i) {
            agg.point_data.push_back(local.pts[i]);
            agg.edge_to_point_map[local.ids[i]] = agg.all_pts + Index(i);
          }
          agg.all_pts += Index(local.ids.size());
        });
  }

  // --- Phase 2: build half-edges from split map ---

  template <typename FaceRange>
  auto build_half_edges_from_map(tf::half_edges<Index> &he,
                                 const FaceRange &face_half_edges,
                                 Index n_original_pts) -> Index {
    _outer_half_edges.allocate(Index(_point_data.size()) * 2);

    struct local_data {
      tf::buffer<he_t> inner_hes;
      tf::buffer<Index> edge_ids;
      Index polygon_offset = 0;
      Index half_edge_offset = 0;
    };
    struct aggregate_data {
      tf::half_edges<Index> &he;
      tf::buffer<he_t> &outer_half_edges;
      tf::buffer<he_t> &inner_half_edges;
      Index polygon_offset = 0;
      Index inner_half_edge_offset = 0;
      Index n_original_half_edges;
      Index n_original_polys;
    };

    Index n_original_half_edges = Index(he.half_edges_buffer().size());
    Index n_original_polys = Index(face_half_edges.size());

    aggregate_data agg{he,
                       _outer_half_edges,
                       _inner_half_edges,
                       0,
                       0,
                       n_original_half_edges,
                       n_original_polys};

    tf::blocked_reduce(
        face_half_edges, agg, local_data{},
        [this, &he, n_original_half_edges, n_original_polys,
         n_original_pts](auto r, local_data &local) {
          for (const auto &heh : r) {
            if (!heh.is_valid())
              continue;
            this->process_triangle(heh, he, local.inner_hes, local.edge_ids,
                             n_original_half_edges, local.half_edge_offset,
                             local.polygon_offset, n_original_polys,
                             n_original_pts);
          }
        },
        [](const local_data &local, aggregate_data &agg) {
          Index n_already_fixed =
              agg.n_original_half_edges +
              Index(agg.outer_half_edges.size());
          // Append local inner HEs to global
          Index inner_before = Index(agg.inner_half_edges.size());
          for (std::size_t j = 0; j < local.inner_hes.size(); ++j)
            agg.inner_half_edges.push_back(local.inner_hes[j]);
          // Fixup IDs
          for (std::size_t j = 0; j < local.edge_ids.size(); ++j) {
            auto id = local.edge_ids[j];
            he_t *hp;
            if (id < agg.n_original_half_edges)
              hp = &agg.he.half_edge(half_edge_handle_t{id});
            else if (id < n_already_fixed)
              hp = &agg.outer_half_edges[id - agg.n_original_half_edges];
            else
              hp = &agg.inner_half_edges[inner_before +
                                         (id - n_already_fixed)];
            if (hp->next >= n_already_fixed)
              hp->next += agg.inner_half_edge_offset;
            if (hp->face >= agg.n_original_polys)
              hp->face += agg.polygon_offset;
          }
          agg.polygon_offset += local.polygon_offset;
          agg.inner_half_edge_offset += local.half_edge_offset;
        });

    return agg.polygon_offset + n_original_polys;
  }

  // --- Phase 3: write computed half-edges to main buffer ---

  auto write_computed_halfedges(tf::half_edges<Index> &he) -> void {
    auto &buf = he.half_edges_buffer();
    auto old_size = buf.size();
    tf::buffer<he_t> tmp;
    tmp.allocate(old_size + _outer_half_edges.size() +
                 _inner_half_edges.size());
    tf::parallel_copy(tf::make_range(buf),
                      tf::take(tf::make_range(tmp), old_size));
    tf::parallel_copy(
        tf::make_range(_outer_half_edges),
        tf::slice(tf::make_range(tmp), old_size,
                  old_size + _outer_half_edges.size()));
    tf::parallel_copy(
        tf::make_range(_inner_half_edges),
        tf::drop(tf::make_range(tmp),
                 old_size + _outer_half_edges.size()));
    buf = std::move(tmp);
  }

  // --- Phase 4: relink boundary edges ---

  auto relink_boundary_edges(tf::half_edges<Index> &he,
                             Index n_original_half_edges) -> void {
    tf::parallel_for_each(
        tf::take(he.half_edge_handles(), n_original_half_edges),
        [&, clean = _clean_tag](auto heh) {
          if (!heh.is_valid())
            return;
          if (!he.is_boundary(tf::unsafe, heh))
            return;
          auto next_heh = he.next(tf::unsafe, heh);
          if (!next_heh.is_valid() || next_heh.id() >= n_original_half_edges)
            return;
          auto eh = he.edge_handle(tf::unsafe, next_heh);
          if (_edge_to_point_map[eh.id()] == clean)
            return;
          he.half_edge(heh).next = _previous_boundary_edge[eh.id()];
        });
  }

  // --- Tessellation helpers ---

  auto compute_tesselation_mask(const tf::half_edges<Index> &he,
                                half_edge_handle_t heh,
                                std::array<half_edge_handle_t, 3> &triangle)
      -> int {
    int mask = 0;
    auto check = [&](int i, half_edge_handle_t h) {
      auto eh = he.edge_handle(tf::unsafe, h);
      if (_edge_to_point_map[eh.id()] != _clean_tag)
        mask |= edge_mask(i);
    };
    check(0, heh);
    auto heh_n = he.next(tf::unsafe, heh);
    check(1, heh_n);
    auto heh_nn = he.next(tf::unsafe, heh_n);
    check(2, heh_nn);
    triangle[0] = heh;
    triangle[1] = heh_n;
    triangle[2] = heh_nn;
    return mask;
  }

  auto process_outer_id(int i, tf::half_edges<Index> &he,
                        const std::array<half_edge_handle_t, 3> &triangle,
                        std::array<Index, 21> &ids,
                        Index n_original_half_edges, Index n_original_pts)
      -> void {
    auto heh = triangle[(i - 3) >> 1];
    bool is_right = (i - 3) & 1;
    auto opposite_heh = he.opposite(tf::unsafe, heh);
    auto eh = he.edge_handle(tf::unsafe, heh);
    auto heh_base = he.half_edge_handle(
        tf::unsafe,
        edge_handle_t{_edge_to_point_map[eh.id()] - n_original_pts}, false);

    if (!he.is_boundary(tf::unsafe, opposite_heh)) {
      if (heh < opposite_heh) {
        if (!is_right) {
          ids[i] = heh.id();
          Index right_id = he.opposite(tf::unsafe, heh_base).id();
          ids[i + 1] = right_id + n_original_half_edges;
          _outer_half_edges[right_id].vertex =
              _edge_to_point_map[eh.id()];
        }
      } else {
        if (is_right) {
          ids[i] = heh.id();
          auto right_vertex = he.half_edge(heh).vertex;
          he.half_edge(heh).vertex = _edge_to_point_map[eh.id()];
          ids[i - 1] = heh_base.id() + n_original_half_edges;
          _outer_half_edges[heh_base.id()].vertex = right_vertex;
        }
      }
    } else if (!is_right) {
      ids[i] = heh.id();
      Index right_id = he.opposite(tf::unsafe, heh_base).id();
      Index mapped_right_id = right_id + n_original_half_edges;
      ids[i + 1] = mapped_right_id;
      _outer_half_edges[right_id].vertex = _edge_to_point_map[eh.id()];
      // boundary handling
      auto opposite_of_left = opposite_heh;
      auto opposite_of_right = heh_base;
      _outer_half_edges[opposite_of_right.id()] =
          he.half_edge(opposite_of_left);
      _outer_half_edges[opposite_of_right.id()].next =
          opposite_of_left.id();
      _previous_boundary_edge[he.edge_handle(tf::unsafe, opposite_of_left)
                                  .id()] =
          opposite_of_right.id() + n_original_half_edges;
      he.half_edge(opposite_of_left).vertex = _edge_to_point_map[eh.id()];
    }
  }

  template <typename InnerBuf>
  auto process_splitting_id(
      int i, const tf::half_edges<Index> &he, InnerBuf &interior_hes,
      const std::array<half_edge_handle_t, 3> &triangle,
      std::array<Index, 21> &ids, Index n_already_fixed_half_edges,
      Index current_hes_offset) -> Index {
    bool start_on_edge = !((i - first_crossing_edge()) & 1);
    if (!start_on_edge)
      return current_hes_offset;
    auto edge_id = (i - first_interior_edge()) >> 1;
    auto cross_vertex = (edge_id + 2) % 3;
    auto heh = triangle[edge_id];
    auto cross_heh = triangle[cross_vertex];
    auto eh = he.edge_handle(tf::unsafe, heh);
    auto v_id = _edge_to_point_map[eh.id()];
    auto id0 = current_hes_offset++;
    auto id1 = current_hes_offset++;
    ids[i] = n_already_fixed_half_edges + id0;
    ids[i + 1] = n_already_fixed_half_edges + id1;
    he_t h0{};
    h0.vertex = v_id;
    interior_hes.push_back(h0);
    he_t h1{};
    h1.vertex = he.start_vertex_handle(tf::unsafe, cross_heh).id();
    interior_hes.push_back(h1);
    return current_hes_offset;
  }

  template <typename InnerBuf>
  auto process_crossing_id(
      int i, const tf::half_edges<Index> &he, InnerBuf &interior_hes,
      const std::array<half_edge_handle_t, 3> &triangle,
      std::array<Index, 21> &ids, Index n_already_fixed_half_edges,
      Index current_hes_offset) -> Index {
    bool start_on_edge = !((i - first_crossing_edge()) & 1);
    if (!start_on_edge)
      return current_hes_offset;
    auto edge_id0 = (i - first_crossing_edge()) >> 1;
    auto edge_id1 = (edge_id0 + 1) % 3;
    auto heh0 = triangle[edge_id0];
    auto eh0 = he.edge_handle(tf::unsafe, heh0);
    auto v_id0 = _edge_to_point_map[eh0.id()];
    auto heh1 = triangle[edge_id1];
    auto eh1 = he.edge_handle(tf::unsafe, heh1);
    auto v_id1 = _edge_to_point_map[eh1.id()];
    auto id0 = current_hes_offset++;
    auto id1 = current_hes_offset++;
    ids[i] = n_already_fixed_half_edges + id0;
    ids[i + 1] = n_already_fixed_half_edges + id1;
    he_t h0{};
    h0.vertex = v_id0;
    interior_hes.push_back(h0);
    he_t h1{};
    h1.vertex = v_id1;
    interior_hes.push_back(h1);
    return current_hes_offset;
  }

  template <typename InnerBuf>
  auto compute_ids(tf::half_edges<Index> &he, InnerBuf &interior_hes,
                   const std::array<half_edge_handle_t, 3> &triangle,
                   std::array<Index, 21> &ids,
                   const std::array<int, 15> &tessalation,
                   Index n_original_half_edges, Index current_hes_offset,
                   Index n_original_pts) -> Index {
    Index n_already_fixed_half_edges =
        n_original_half_edges + Index(_point_data.size()) * 2;
    int n_entries = 3 * tessalation[0];
    for (int idx = 0; idx < n_entries; ++idx) {
      int entry = tessalation[3 + idx];
      if (entry < first_split_edge())
        ids[entry] = triangle[entry].id();
      else if (entry < first_interior_edge())
        process_outer_id(entry, he, triangle, ids, n_original_half_edges,
                         n_original_pts);
      else if (entry < first_crossing_edge())
        current_hes_offset = process_splitting_id(
            entry, he, interior_hes, triangle, ids,
            n_already_fixed_half_edges, current_hes_offset);
      else
        current_hes_offset = process_crossing_id(
            entry, he, interior_hes, triangle, ids,
            n_already_fixed_half_edges, current_hes_offset);
    }
    return current_hes_offset;
  }

  template <typename InnerBuf>
  auto get_half_edge(Index id, tf::half_edges<Index> &he,
                     InnerBuf &interior_hes, Index n_original_half_edges,
                     Index n_already_fixed_half_edges) -> he_t & {
    if (id < n_original_half_edges)
      return he.half_edge(half_edge_handle_t{id});
    else if (id < n_already_fixed_half_edges)
      return _outer_half_edges[id - n_original_half_edges];
    else
      return interior_hes[id - n_already_fixed_half_edges];
  }

  template <typename InnerBuf>
  auto process_triangle(half_edge_handle_t heh, tf::half_edges<Index> &he,
                        InnerBuf &interior_hes,
                        tf::buffer<Index> &all_ids,
                        Index n_original_half_edges,
                        Index &current_hes_offset,
                        Index &current_polygon_offset,
                        Index n_original_polys, Index n_original_pts)
      -> void {
    std::array<half_edge_handle_t, 3> triangle;
    std::array<Index, 21> ids;
    int case_mask = compute_tesselation_mask(he, heh, triangle);
    if (case_mask == 0)
      return;
    Index n_already_fixed_half_edges =
        n_original_half_edges + Index(_outer_half_edges.size());
    const auto &tess = tessalation_case(case_mask);
    current_hes_offset =
        compute_ids(he, interior_hes, triangle, ids, tess,
                    n_original_half_edges, current_hes_offset, n_original_pts);

    auto process_subtriangle = [&](int base, Index face_id) {
      Index he_ids[3] = {ids[tess[base]], ids[tess[base + 1]],
                         ids[tess[base + 2]]};
      all_ids.push_back(he_ids[0]);
      all_ids.push_back(he_ids[1]);
      all_ids.push_back(he_ids[2]);
      auto &h0 = get_half_edge(he_ids[0], he, interior_hes,
                                n_original_half_edges,
                                n_already_fixed_half_edges);
      auto &h1 = get_half_edge(he_ids[1], he, interior_hes,
                                n_original_half_edges,
                                n_already_fixed_half_edges);
      auto &h2 = get_half_edge(he_ids[2], he, interior_hes,
                                n_original_half_edges,
                                n_already_fixed_half_edges);
      h0.next = he_ids[1];
      h1.next = he_ids[2];
      h2.next = he_ids[0];
      h0.face = face_id;
      h1.face = face_id;
      h2.face = face_id;
    };

    // First sub-triangle reuses the original face
    process_subtriangle(3, he.half_edge(triangle[0]).face);
    // Remaining sub-triangles get new face IDs
    for (int t = 1; t < tess[0]; ++t)
      process_subtriangle(3 + t * 3,
                          n_original_polys + current_polygon_offset++);
  }

  // --- Static helpers ---

  static constexpr auto edge_mask(int i) -> int {
    constexpr int masks[3] = {1, 2, 4};
    return masks[i];
  }

  static auto tessalation_case(int i) -> const std::array<int, 15> & {
    static constexpr std::array<std::array<int, 15>, 16> cases = {{
        {{1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{2, 2, 2, 3, 9, 2, 4, 1, 10, 0, 0, 0, 0, 0, 0}},
        {{2, 2, 2, 0, 5, 11, 6, 2, 12, 0, 0, 0, 0, 0, 0}},
        {{3, 4, 4, 4, 5, 16, 15, 6, 10, 2, 3, 9, 0, 0, 0}},
        {{2, 2, 2, 0, 14, 8, 13, 1, 7, 0, 0, 0, 0, 0, 0}},
        {{3, 4, 4, 3, 20, 8, 19, 4, 14, 1, 7, 13, 0, 0, 0}},
        {{3, 4, 4, 18, 6, 7, 0, 5, 11, 17, 8, 12, 0, 0, 0}},
        {{4, 6, 6, 3, 20, 8, 4, 5, 16, 19, 15, 17, 18, 6, 7}},
        {{1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{2, 2, 2, 3, 9, 2, 4, 1, 10, 0, 0, 0, 0, 0, 0}},
        {{2, 2, 2, 0, 5, 11, 6, 2, 12, 0, 0, 0, 0, 0, 0}},
        {{3, 4, 4, 4, 5, 16, 3, 15, 11, 6, 2, 12, 0, 0, 0}},
        {{2, 2, 2, 0, 14, 8, 13, 1, 7, 0, 0, 0, 0, 0, 0}},
        {{3, 4, 4, 3, 20, 8, 4, 1, 10, 7, 19, 9, 0, 0, 0}},
        {{3, 4, 4, 6, 7, 18, 8, 0, 14, 5, 17, 13, 0, 0, 0}},
        {{4, 6, 6, 3, 20, 8, 4, 5, 16, 19, 15, 17, 18, 6, 7}},
    }};
    return cases[i];
  }

  static constexpr auto first_split_edge() -> int { return 3; }
  static constexpr auto first_interior_edge() -> int { return 9; }
  static constexpr auto first_crossing_edge() -> int { return 15; }

  // --- Members ---

  static constexpr Index _clean_tag = Index(-1);
  tf::buffer<he_t> _outer_half_edges;
  tf::buffer<he_t> _inner_half_edges;
  tf::points_buffer<Real, Dims> _point_data;
  tf::buffer<Index> _edge_to_point_map;
  tf::buffer<Index> _previous_boundary_edge;
};

} // namespace remesh
} // namespace tf
