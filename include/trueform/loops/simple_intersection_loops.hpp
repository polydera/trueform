/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/buffer.hpp"
#include "../core/polygons.hpp"
#include "../core/views/block_indirect_range.hpp"
#include "../intersect/base/simple_intersections.hpp"
#include "./loop_extractor.hpp"
#include "./vertex.hpp"

namespace tf::loop {
template <typename Index> class simple_intersection_loops {
public:
  template <typename Policy, typename RealT, std::size_t Dims>
  auto
  build(const tf::polygons<Policy> &polygons,
        const tf::intersect::simple_intersections<Index, RealT, Dims> &si) {
    clear();
    initialize(polygons.points().size(), si.intersection_points().size());
    Index offset = 0;
    auto result = std::tie(_loop_vertices, _loop_offsets, _vertices);
    auto local_result =
        std::make_tuple(tf::buffer<vertex<Index>>{}, tf::buffer<Index>{},
                        loop_extractor<Index, RealT>{});

    auto task_f = [&](const auto &r, auto &tup) {
      auto &[loop_vertices, loop_offsets, extractor] = tup;
      for (const auto &intersections : r) {
        extractor.build(polygons.faces()[intersections.front().polygon],
                        si.intersection_points(), polygons.points(),
                        intersections);
        extractor.extract(loop_offsets, loop_vertices);
      }
    };

    auto aggregate_f = [&](const auto &local_result, auto &result) {
      const auto &[l_loop_vertices, l_loop_offsets, _] = local_result;
      (void)_; // suppress unused warning
      auto &[loop_vertices, loop_offsets, vertices] = result;
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
      for (auto vertex : l_loop_vertices) {
        if (vertex.source == tf::loop::vertex_source::original) {
          if (_map[vertex.id] == -1) {
            _map[vertex.id] = _map_offset++;
            vertices.push_back(vertex);
          }
          *it_loop_vertices++ = _map[vertex.id];
        } else {
          *it_loop_vertices++ = vertex.id;
        }
      }
    };

    tf::blocked_reduce_sequenced_aggregate(si.intersections(), result,
                                           local_result, task_f, aggregate_f);
    if (_loop_offsets.size())
      _loop_offsets.push_back(_loop_vertices.size());
  }

  auto loops() const {
    return tf::make_offset_block_range(_loop_offsets, _loop_vertices);
  }

  auto mapped_loops() const {
    return tf::make_block_indirect_range(loops(), _vertices);
  }

  auto vertices() const -> const tf::buffer<vertex<Index>> & {
    return _vertices;
  }

  auto clear() {
    _loop_vertices.clear();
    _loop_offsets.clear();
    _vertices.clear();
    _map.clear();
    _map_offset = 0;
  }

private:
  auto initialize(std::size_t n_points, std::size_t n_intersection_points) {
    _map_offset = n_intersection_points;
    _map.allocate(n_points);
    tf::parallel_fill(_map, -1);
  }
  Index _map_offset;
  tf::buffer<Index> _loop_vertices;
  tf::buffer<Index> _loop_offsets;
  tf::buffer<vertex<Index>> _vertices;
  tf::buffer<Index> _map;
};
} // namespace tf::loop
