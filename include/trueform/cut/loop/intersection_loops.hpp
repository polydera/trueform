/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/enumerate.hpp"
#include "./loop_extractor.hpp"
#include "./vertex.hpp"

namespace tf::loop {
template <typename Index, typename ObjectKey> class intersection_loops {
public:
  auto loops() const {
    return tf::make_offset_block_range(_loop_offsets, _loop_vertices);
  }

  auto descriptors() const { return tf::make_range(_object_keys); }

  auto mapped_loops() const {
    return tf::make_block_indirect_range(loops(), _vertices);
  }

  auto vertices() const -> const tf::buffer<vertex<Index>> & {
    return _vertices;
  }

  auto clear() {
    _loop_vertices.clear();
    _loop_offsets.clear();
    _object_keys.clear();
    _vertices.clear();
  }

protected:
  auto initialize(std::size_t n_intersection_points) {
    _vertices.allocate(n_intersection_points);
    tf::parallel_apply(tf::enumerate(_vertices), [](auto &&pair) {
      auto &&[id, v] = pair;
      v = {Index(id), vertex_source::created};
    });
  }

  template <typename Range, typename Policy1, typename F0, typename F1>
  auto build(const Range &intersections,
             const tf::points<Policy1> &intersection_points,
             const F0 &apply_to_polygons, const F1 &handle_id) {
    clear();
    initialize(intersection_points.size());
    Index offset = 0;
    auto result =
        std::tie(_object_keys, _loop_vertices, _loop_offsets, _vertices);
    auto local_result =
        std::make_tuple(tf::buffer<ObjectKey>{}, tf::buffer<vertex<Index>>{},
                        tf::buffer<Index>{},
                        loop_extractor<Index, tf::coordinate_type<Policy1>>{});

    auto task_f = [&](const auto &r, auto &tup) {
      auto &[object_keys, loop_vertices, loop_offsets, extractor] = tup;
      for (const auto &intersections : r) {
        apply_to_polygons(
            intersections.front(),
            [&object_keys = object_keys, &loop_vertices = loop_vertices,
             &loop_offsets = loop_offsets, &intersections = intersections,
             &intersection_points = intersection_points,
             &extractor = extractor](const auto &polygons) {
              Index n_loops = extractor.build(
                  polygons.faces()[intersections.front().object],
                  intersection_points,
                  polygons.points() | tf::tag(tf::frame_of(polygons)),
                  intersections, loop_offsets, loop_vertices);
              ObjectKey key{intersections.front().object_key()};

              for (Index i = 0; i < n_loops; ++i)
                object_keys.push_back(key);
            });
      }
    };

    auto aggregate_f = [&](const auto &local_result, auto &result) {
      const auto &[l_object_keys, l_loop_vertices, l_loop_offsets, _] =
          local_result;
      (void)_; // suppress unused warning
      auto &[object_keys, loop_vertices, loop_offsets, vertices] = result;
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

      auto copy_loop_f = [&, &vertices = vertices](auto key, auto end) {
        while (l_it_vertices != end) {
          auto vertex = *l_it_vertices++;
          auto [is_new, id] = handle_id(key, vertex);
          if (is_new)
            vertices.push_back(vertex);
          *it_loop_vertices++ = id;
        }
      };

      for (auto [key, e] : tf::zip(
               tf::make_range(l_object_keys.begin(), l_object_keys.end() - 1),
               tf::make_range(l_loop_offsets.begin() + 1,
                              l_loop_offsets.end())))
        copy_loop_f(key, l_loop_vertices.begin() + e);
      copy_loop_f(l_object_keys.back(), l_loop_vertices.end());
    };

    tf::blocked_reduce_sequenced_aggregate(intersections, result, local_result,
                                           task_f, aggregate_f);
    if (_loop_offsets.size())
      _loop_offsets.push_back(_loop_vertices.size());
  }

  tf::buffer<Index> _loop_vertices;
  tf::buffer<Index> _loop_offsets;
  tf::buffer<vertex<Index>> _vertices;
  tf::buffer<ObjectKey> _object_keys;
};
} // namespace tf::loop
