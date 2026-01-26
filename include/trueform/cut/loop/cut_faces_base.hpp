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
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/array_hash.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_set.hpp"
#include "../../core/plane.hpp"
#include "../../core/transformed.hpp"
#include "./loop_extractor.hpp"
#include "./vertex.hpp"

namespace tf::loop {
template <typename Index, typename ObjectKey> class cut_faces_base {
public:
  auto loops() const {
    return tf::make_offset_block_range(_loop_offsets, _loop_vertices);
  }

  auto descriptors() const { return tf::make_range(_object_keys); }

  auto mapped_loops() const {
    return tf::make_offset_block_range(_loop_offsets, _vertices);
  }

  auto vertices() const -> const tf::buffer<vertex<Index>> & {
    return _vertices;
  }

  auto intersection_edges() const {
    return tf::make_range(_intersection_edges);
  }

  auto is_intersection_edge(vertex<Index> v0, vertex<Index> v1) const {
    if (v1 < v0)
      std::swap(v0, v1);
    if (v0.source != tf::loop::vertex_source::created ||
        v1.source != tf::loop::vertex_source::created)
      return false;
    return _intersection_edges_set.find(std::array<Index, 2>{v0.id, v1.id}) !=
           _intersection_edges_set.end();
  }

  auto clear() {
    _loop_vertices.clear();
    _loop_offsets.clear();
    _object_keys.clear();
    _vertices.clear();
    _intersection_edges.clear();
    _intersection_edges_set.clear();
  }

protected:
  template <typename Range, typename Policy1, typename F0, typename F1,
            typename F2>
  auto build(const Range &intersections,
             const tf::points<Policy1> &intersection_points,
             const F0 &apply_to_polygons, const F1 &handle_id,
             const F2 &get_flat_id) {
    clear();
    Index offset = 0;
    Index total_vertices = intersection_points.size();
    auto result =
        std::tie(_object_keys, _loop_vertices, _loop_offsets, _vertices,
                 _intersection_edges, _intersection_edges_set);
    auto local_result = std::make_tuple(
        tf::buffer<ObjectKey>{}, tf::buffer<vertex<Index>>{},
        tf::buffer<Index>{}, tf::blocked_buffer<vertex<Index>, 2>{},
        loop_extractor<Index, tf::coordinate_type<Policy1>>{});

    auto task_f = [&](const auto &r, auto &tup) {
      for (const auto &intersections : r) {
        /*if(intersections.front().object != 500559)*/
        /*  continue;*/
        apply_to_polygons(intersections.front(),
                          [&](const auto &polygons, const auto &polygons_other) {
          auto &[object_keys, loop_vertices, loop_offsets, intersection_edges,
                 extractor] = tup;
          auto object = intersections.front().object;
          auto frame = tf::frame_of(polygons);
          auto frame_other = tf::frame_of(polygons_other);
          auto plane =
              tf::transformed(tf::make_plane(polygons[object]), frame);

          auto describe_other = [&](auto object_other) {
            auto plane_other = tf::transformed(
                tf::make_plane(polygons_other[object_other]), frame_other);

            auto cross = tf::cross(plane.normal, plane_other.normal);
            auto cross_len_sq = tf::dot(cross, cross);
            using RealT = std::decay_t<decltype(cross_len_sq)>;
            constexpr auto eps = tf::epsilon<RealT>;

            if (cross_len_sq > eps * eps)
              return std::make_pair(false, std::size_t(0));

            auto n_dot = tf::dot(plane.normal, plane_other.normal);
            if (std::abs(plane.d - plane_other.d * n_dot) > eps)
              return std::make_pair(false, std::size_t(0));

            return std::make_pair(true, polygons_other.faces()[object_other].size());
          };

          Index n_loops = extractor.build(
              polygons.faces()[object], intersection_points,
              polygons.points() | tf::tag(frame), intersections, get_flat_id,
              describe_other, loop_offsets, loop_vertices);
          ObjectKey key{intersections.front().object_key()};
          for (const auto &edge : extractor.intersection_edges()) {
            intersection_edges.emplace_back(edge[0], edge[1]);
          }

          for (Index i = 0; i < n_loops; ++i)
            object_keys.push_back(key);
        });
      }
    };

    auto aggregate_f = [&](const auto &local_result, auto &result) {
      const auto &[l_object_keys, l_loop_vertices, l_loop_offsets,
                   l_intersection_edges, _] = local_result;
      if (l_object_keys.size() == 0)
        return;
      (void)_; // suppress unused warning
      auto &[object_keys, loop_vertices, loop_offsets, vertices,
             intersection_edges, intersection_edges_set] = result;
      //
      auto old_ids_size = object_keys.size();
      object_keys.reallocate(old_ids_size + l_object_keys.size());
      std::copy(l_object_keys.begin(), l_object_keys.end(),
                object_keys.begin() + old_ids_size);
      //
      auto old_offsets_size = loop_offsets.size();
      loop_offsets.reallocate(old_offsets_size + l_loop_offsets.size());
      auto it_offsets = loop_offsets.begin() + old_offsets_size;
      for (auto e : l_loop_offsets)
        *it_offsets++ = offset + e;
      offset += l_loop_vertices.size();
      //
      auto old_loop_vertices_size = loop_vertices.size();
      loop_vertices.reallocate(old_loop_vertices_size + l_loop_vertices.size());
      auto it_loop_vertices = loop_vertices.begin() + old_loop_vertices_size;
      auto l_it_vertices = l_loop_vertices.begin();
      //
      auto old_vertices_size = vertices.size();
      vertices.reallocate(old_vertices_size + l_loop_vertices.size());
      auto it_loop_vertices2 = vertices.begin() + old_vertices_size;

      auto copy_loop_f = [&](auto key, auto end) {
        while (l_it_vertices != end) {
          auto vertex = *l_it_vertices++;
          auto [is_new, id] = handle_id(key, vertex);
          total_vertices += is_new;
          *it_loop_vertices++ = id;
          *it_loop_vertices2++ = vertex;
        }
      };

      for (auto [key, e] : tf::zip(
               tf::make_range(l_object_keys.begin(), l_object_keys.end() - 1),
               tf::make_range(l_loop_offsets.begin() + 1,
                              l_loop_offsets.end())))
        copy_loop_f(key, l_loop_vertices.begin() + e);
      copy_loop_f(l_object_keys.back(), l_loop_vertices.end());

      for (const auto &edge : l_intersection_edges) {
        if (intersection_edges_set.insert({edge[0].id, edge[1].id}).second)
          intersection_edges.emplace_back(edge[0], edge[1]);
      }
    };

    tf::blocked_reduce_sequenced_aggregate(intersections, result, local_result,
                                           task_f, aggregate_f);
    if (_loop_offsets.size())
      _loop_offsets.push_back(_loop_vertices.size());
    return total_vertices;
  }

  tf::hash_set<std::array<Index, 2>, tf::array_hash<Index, 2>>
      _intersection_edges_set;
  tf::blocked_buffer<vertex<Index>, 2> _intersection_edges;
  tf::buffer<Index> _loop_vertices;
  tf::buffer<Index> _loop_offsets;
  tf::buffer<vertex<Index>> _vertices;
  tf::buffer<ObjectKey> _object_keys;
};
} // namespace tf::loop
