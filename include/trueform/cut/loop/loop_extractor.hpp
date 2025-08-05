/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/buffer.hpp"
#include "../../core/dot.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/normal.hpp"
#include "../../core/points.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../intersect/simple_intersection.hpp"
#include "../../intersect/tagged_intersection.hpp"
#include "./cut_face_by_intersections.hpp"
#include "./vertex.hpp"
#include <algorithm>

inline float time_tf0=0;
namespace tf::loop {
template <typename Index, typename RealT> class loop_extractor {

public:
  template <typename Range0, typename Policy0, typename Policy, typename Range2>
  auto build(const Range0 &face, const tf::points<Policy0> &intersection_points,
             const tf::points<Policy> &mesh_points, const Range2 &intersections,
             tf::buffer<Index> &offsets, tf::buffer<vertex<Index>> &vertices) {
    if (is_simple_case(intersections)) {
      return extract_simple_case(face, intersections, offsets, vertices);
    }
    clear();
    extract_base_loop_from_intersections(face, intersection_points, mesh_points,
                                         intersections);
    build_base_loop_edges();
    extract_edges(intersections, intersection_points, intersections[0]);
    return extract(face, intersection_points, mesh_points, offsets, vertices);
  }

  auto clear() {
    _edges.clear();
    _work_buffer.clear();
    _base_loop.clear();
    _base_loop_edges.clear();
    _cf.clear();
  }

  auto base_loop() const -> const tf::buffer<vertex<Index>> & {
    return _base_loop;
  }

  auto edges() const { return tf::make_blocked_range<2>(_edges); }

private:
  template <typename Range0, typename Range>
  auto extract_simple_case(const Range0 &face, const Range &r,
                           tf::buffer<Index> &offsets,
                           tf::buffer<vertex<Index>> &vertices) {
    auto i0 = r[0];
    auto i1 = r[1];
    if (i0.target.id > i1.target.id)
      std::swap(i0, i1);

    offsets.push_back(vertices.size());
    for (Index i = 0; i <= Index(i0.target.id); ++i)
      vertices.push_back({Index(face[i]), vertex_source::original});
    vertices.push_back({i0.id, vertex_source::created});
    vertices.push_back({i1.id, vertex_source::created});

    for (Index i = i1.target.id + 1; i < Index(face.size()); ++i)
      vertices.push_back({Index(face[i]), vertex_source::original});

    // add second loop
    offsets.push_back(vertices.size());
    vertices.push_back({i1.id, vertex_source::created});
    vertices.push_back({i0.id, vertex_source::created});
    for (Index i = i0.target.id + 1; i <= i1.target.id; ++i)
      vertices.push_back({Index(face[i]), vertex_source::original});
    return Index(2);
  }

  template <typename Range> auto is_simple_case(const Range &r) {
    return r.size() == 2 && r[0].target.label == tf::topo_type::edge &&
           r[1].target.label == tf::topo_type::edge &&
           r[0].target.id != r[1].target.id;
  }

  template <typename Range0, typename Policy0, typename Policy>
  auto
  extract(const Range0 &face, const tf::points<Policy0> &intersection_points,
          const tf::points<Policy> &mesh_points, tf::buffer<Index> &offsets,
          tf::buffer<vertex<Index>> &vertices) {
    if (edges().size() == 0) {
      offsets.push_back(vertices.size());
      std::copy(base_loop().begin(), base_loop().end(),
                std::back_inserter(vertices));
      return Index(1);
    } else {
      return extract_generic(face, intersection_points, mesh_points, offsets,
                             vertices);
    }
    return Index(0);
  }

  template <typename Range0, typename Policy0, typename Policy>
  auto extract_generic(const Range0 &face,
                       const tf::points<Policy0> &intersection_points,
                       const tf::points<Policy> &mesh_points,
                       tf::buffer<Index> &offsets,
                       tf::buffer<vertex<Index>> &vertices) {

    const auto &frame = tf::frame_of(mesh_points);
    auto projector = tf::make_simple_projector(tf::transformed_normal(
        tf::make_normal(tf::make_polygon(face, mesh_points)), frame));
    _cf.build(base_loop(), tf::make_edges(edges()),
              [&](const vertex<Index> &v) -> tf::point<RealT, 2> {
                if (v.source == vertex_source::created)
                  return projector(intersection_points[v.id]);
                else
                  return projector(tf::transformed(mesh_points[v.id], frame));
              });

    Index old_size = offsets.size();
    _cf.extract(offsets, vertices);
    return Index(offsets.size()) - old_size;
  }

  auto build_base_loop_edges() {
    Index size = _base_loop.size();
    Index prev = size - 1;
    for (Index i = 0; i < size; prev = i++) {
      if (_base_loop[prev] < _base_loop[i])
        _base_loop_edges.push_back({_base_loop[prev], _base_loop[i]});
      else
        _base_loop_edges.push_back({_base_loop[i], _base_loop[prev]});
    }
    std::sort(_base_loop_edges.begin(), _base_loop_edges.end());
  }

  auto should_add_edge(const vertex<Index> &v0, const vertex<Index> &v1) {
    if (v0 == v1)
      return false;
    auto edge = std::array<vertex<Index>, 2>{v0, v1};
    auto it = std::lower_bound(_base_loop_edges.begin(), _base_loop_edges.end(),
                               edge);
    if (it != _base_loop_edges.end() && *it == edge)
      return false;
    return true;
  }

  auto add_edge(vertex<Index> v0, vertex<Index> v1) {
    if (v1 < v0)
      std::swap(v0, v1);
    if (should_add_edge(v0, v1)) {
      _edges.push_back(v0);
      _edges.push_back(v1);
    }
  }

  template <typename Range, typename Policy>
  auto extract_edges(const Range &intersections, const tf::points<Policy> &,
                     tf::intersect::simple_intersection<Index>) {
    if (intersections.size() != 2)
      return;

    add_edge({intersections[0].id, vertex_source::created},
             {intersections[1].id, vertex_source::created});
  }

  template <typename Range, typename Policy>
  auto extract_edges(const Range &intersections,
                     const tf::points<Policy> &intersection_points,
                     tf::intersect::tagged_intersection<Index>) {
    auto it = intersections.begin();
    auto end = intersections.end();
    while (it != end) {
      auto next = std::find_if(it + 1, end, [it](const auto &x) {
        return x.object_other != it->object_other;
      });
      if (next - it == 2) {
        add_edge({it->id, vertex_source::created},
                 {(it + 1)->id, vertex_source::created});
      } else if (next - it > 2) {
        extract_edges(it, next, intersection_points);
      }
      it = next;
    }
    make_edges_unique();
  }

  template <typename Iterator, typename Policy>
  auto extract_edges(Iterator begin, Iterator end,
                     const tf::points<Policy> &intersection_points) {
    std::array<std::pair<RealT, Iterator>,
               tf::static_size_v<decltype(intersection_points.front())>>
        min;
    min.fill({std::numeric_limits<RealT>::max(), begin});
    std::array<std::pair<RealT, Iterator>,
               tf::static_size_v<decltype(intersection_points.front())>>
        max;
    max.fill({std::numeric_limits<RealT>::min(), begin});

    _work_buffer.clear();
    auto it = begin;
    while (it != end) {
      auto pt = intersection_points[it->id];
      _work_buffer.push_back({it->target, it->id, RealT(0)});
      for (std::size_t i = 0; i < min.size(); ++i) {
        min[i] = std::min(min[i], std::make_pair(pt[i], it));
        max[i] = std::max(max[i], std::make_pair(pt[i], it));
      }
      ++it;
    }
    auto res = std::make_pair(max[0].first - min[0].first, std::size_t(0));
    for (std::size_t i = 1; i < min.size(); ++i) {
      res = std::max(res, std::make_pair(max[i].first - min[i].first, i));
    }
    auto origin = intersection_points[min[res.second].second->id];
    auto dir = intersection_points[max[res.second].second->id] - origin;

    for (auto &e : _work_buffer)
      e.t = tf::dot(intersection_points[e.id] - origin, dir);
    std::sort(_work_buffer.begin(), _work_buffer.end(),
              [](const auto &x, const auto &y) {
                return std::make_pair(x.t, x.id) < std::make_pair(y.t, y.id);
              });
    for (auto [a, b] : tf::make_slide_range<2>(_work_buffer))
      add_edge({a.id, vertex_source::created}, {b.id, vertex_source::created});
  }

  auto make_edges_unique() {
    auto es = tf::make_blocked_range<2>(_edges);
    std::sort(es.begin(), es.end());
    auto n = (std::unique(es.begin(), es.end()) - es.begin()) * 2;
    _edges.erase_till_end(_edges.begin() + n);
  }

  template <typename Range0, typename Range1, typename Policy, typename Range2>
  auto extract_base_loop_from_intersections(
      const Range0 &face, const Range1 &intersection_points,
      const tf::points<Policy> &mesh_points, const Range2 &intersections) {
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
                                     auto origin, auto edge_dir) {
      while (it != end) {
        if (it->target.id != edge_id)
          return it;
        it->t = tf::dot(edge_dir, intersection_points[it->id] - origin);
        ++it;
      }
      return it;
    };
    Index size = face.size();
    auto it = _work_buffer.begin();
    auto end = _work_buffer.end();
    const auto &frame = tf::frame_of(mesh_points);
    for (Index i = 0; i < size; ++i) {
      Index next = (i + 1) * ((i + 1) < size);
      auto pt0 = tf::transformed(mesh_points[face[i]], frame);
      auto pt1 = tf::transformed(mesh_points[face[next]], frame);
      auto edge_dir = pt1 - pt0;
      auto next_it = find_and_fill_on_edge(it, end, i, pt0, edge_dir);
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
  tf::buffer<std::array<vertex<Index>, 2>> _base_loop_edges;
  tf::loop::cut_face_by_intersections<Index, RealT> _cf;
};

} // namespace tf::loop
