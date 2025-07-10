/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/buffer.hpp"
#include "../core/dot.hpp"
#include "../core/views/blocked_range.hpp"
#include "../intersect/intersection.hpp"
#include "../intersect/simple_intersection.hpp"
#include "./simple_loop_split.hpp"
#include "./vertex.hpp"
#include <algorithm>

namespace tf::loop {
template <typename Index, typename RealT> class loop_extractor {

public:
  template <typename Range0, typename Range1, typename Range2, typename Range3>
  auto build(const Range0 &face, const Range1 &intersection_points,
             const Range2 &mesh_points, const Range3 &intersections) {
    clear();
    extract_base_loop_from_intersections(face, intersection_points, mesh_points,
                                         intersections);
    extract_edges(intersections, intersections[0]);
  }

  auto extract(tf::buffer<Index> &offsets,
               tf::buffer<vertex<Index>> &vertices) {
    if (edges().size() == 0) {
      offsets.push_back(vertices.size());
      std::copy(base_loop().begin(), base_loop().end(),
                std::back_inserter(vertices));

    } else if (edges().size() == 1)
      tf::simple_loop_split(base_loop(),
                            std::array<vertex<Index>, 2>{_edges[0], _edges[1]},
                            offsets, vertices);
  }

  auto clear() {
    _edges.clear();
    _work_buffer.clear();
    _base_loop.clear();
  }

  auto base_loop() const -> const tf::buffer<vertex<Index>> & {
    return _base_loop;
  }

  auto edges() const { return tf::make_blocked_range<2>(_edges); }

private:
  template <typename Range>
  auto extract_edges(const Range &intersections,
                     tf::intersect::simple_intersection<Index>) {
    if (intersections.size() != 2)
      return;
    _edges.push_back({intersections[0].id, vertex_source::created});
    _edges.push_back({intersections[1].id, vertex_source::created});
  }

  template <typename Range>
  auto extract_edges(const Range &intersections,
                     tf::intersect::intersection<Index>) {

    auto it = intersections.begin();
    auto end = intersections.end();
    while (it != end) {
      auto next = std::find_if(it + 1, end, [it](const auto &x) {
        return x.polygon_other != it->polygon_other;
      });
      // each polygon has 1 or 2 intersections with another
      // convex polygon
      if (next - it == 2) {
        _edges.push_back({it->id, vertex_source::created});
        _edges.push_back({(it + 1)->id, vertex_source::created});
      }
      it = next;
    }
  }

  template <typename Range0, typename Range1, typename Range2, typename Range3>
  auto extract_base_loop_from_intersections(const Range0 &face,
                                            const Range1 &intersection_points,
                                            const Range2 &mesh_points,
                                            const Range3 &intersections) {
    for (const auto &x : intersections)
      if (x.target.label != tf::topo_type::face)
        _work_buffer.push_back({x.target, x.id, RealT(0)});
    // (vertex:0|edge:1, sub_id)
    // all vertices will appear before edges
    std::sort(_work_buffer.begin(), _work_buffer.end(),
              [](const auto &x, const auto &y) {
                return std::make_pair(x.target.id, x.target.label) <
                       std::make_pair(y.target.id, y.target.label);
              });
    auto find_and_fill_on_edge = [&](auto it, auto end, Index edge_id,
                                     auto edge_dir) {
      while (it != end) {
        if (it->target.id != edge_id)
          return it;
        it->t = tf::dot(edge_dir, intersection_points[it->id]);
        ++it;
      }
      return it;
    };
    Index size = face.size();
    auto it = _work_buffer.begin();
    auto end = _work_buffer.end();
    for (Index i = 0; i < size; ++i) {
      Index next = (i + 1) * ((i + 1) < size);
      auto edge_dir = mesh_points[face[next]] - mesh_points[face[i]];
      auto next_it = find_and_fill_on_edge(it, end, i, edge_dir);
      // no points on this edge
      if (it == next_it) {
        _base_loop.push_back({Index(face[i]), vertex_source::original});
        continue;
      }
      std::sort(it, next_it, [](const auto &x, const auto &y) {
        // to ensure same ordering of multiple ids with same t
        // and keep vertices before edge points
        return std::make_tuple(x.target.label, x.t, x.id) <
               std::make_tuple(y.target.label, y.t, y.id);
      });
      // configurations exist where we might get duplicates
      auto it_end = std::unique(it, next_it, [](const auto &x, const auto &y) {
        return x.id == y.id;
      });
      if (it != it_end && it->target.label != tf::topo_type::vertex)
        _base_loop.push_back({Index(face[i]), vertex_source::original});
      while (it != it_end)
        _base_loop.push_back({it++->id, vertex_source::created});
      it = next_it;
    }
  }

  struct node_t {
    tf::intersect::intersection_target<Index> target;
    Index id;
    RealT t;
  };

  tf::buffer<node_t> _work_buffer;
  tf::buffer<vertex<Index>> _base_loop;
  tf::buffer<vertex<Index>> _edges;
};

} // namespace tf::loop
