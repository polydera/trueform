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
#include "../../core/dot.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/normal.hpp"
#include "../../core/points.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../intersect/types/intersection.hpp"
#include "../../intersect/types/simple_intersection.hpp"
#include "../../intersect/types/tagged_intersection.hpp"
#include "./cut_face_by_intersections.hpp"
#include "./vertex.hpp"
#include <algorithm>

namespace tf::loop {
template <typename Index, typename RealT> class loop_extractor {

public:
  template <typename Range0, typename Policy0, typename Policy, typename Range2,
            typename F0, typename F1>
  auto build(const Range0 &face, const tf::points<Policy0> &intersection_points,
             const tf::points<Policy> &mesh_points, const Range2 &intersections,
             const F0 &get_flat_id, const F1 &describe_other,
             tf::buffer<Index> &offsets, tf::buffer<vertex<Index>> &vertices) {
    clear();
    if (is_simple_case(intersections)) {
      return extract_simple_case(face, intersections, get_flat_id, offsets,
                                 vertices);
    }
    extract_base_loop_from_intersections(face, intersection_points, mesh_points,
                                         intersections, get_flat_id);
    build_base_loop_edges();
    extract_edges(intersections, intersection_points, intersections[0],
                  get_flat_id, describe_other);
    return extract(face, intersection_points, mesh_points, offsets, vertices);
  }

  template <typename Range0, typename Policy0, typename Policy, typename Range2,
            typename F>
  auto build(const Range0 &face, const tf::points<Policy0> &intersection_points,
             const tf::points<Policy> &mesh_points, const Range2 &intersections,
             const F &get_flat_id, tf::buffer<Index> &offsets,
             tf::buffer<vertex<Index>> &vertices) {
    return build(
        face, intersection_points, mesh_points, intersections, get_flat_id,
        [](const auto &) { return std::make_pair(false, -1); }, offsets,
        vertices);
  }

  auto clear() {
    _edges.clear();
    _work_buffer.clear();
    _base_loop.clear();
    _base_loop_edges.clear();
    _cf.clear();
    _all_edges.clear();
  }

  auto intersection_edges() const {
    return tf::make_blocked_range<2>(_all_edges);
  }

private:
  auto base_loop() const -> const tf::buffer<vertex<Index>> & {
    return _base_loop;
  }

  auto edges() const { return tf::make_blocked_range<2>(_edges); }

  template <typename Range0, typename Range, typename F>
  auto extract_simple_case(const Range0 &face, const Range &r,
                           const F &get_flat_id,

                           tf::buffer<Index> &offsets,
                           tf::buffer<vertex<Index>> &vertices) {
    auto i0 = std::make_pair(r[0], get_flat_id(r[0]));
    auto i1 = std::make_pair(r[1], get_flat_id(r[1]));
    if (i0.first.target.id > i1.first.target.id)
      std::swap(i0, i1);

    offsets.push_back(vertices.size());
    for (Index i = 0; i <= Index(i0.first.target.id); ++i)
      vertices.push_back({Index(face[i]), i, vertex_source::original});
    vertices.push_back({i0.first.id, i0.second, vertex_source::created});
    vertices.push_back({i1.first.id, i1.second, vertex_source::created});

    for (Index i = i1.first.target.id + 1; i < Index(face.size()); ++i)
      vertices.push_back({Index(face[i]), i, vertex_source::original});

    // add second loop
    offsets.push_back(vertices.size());
    vertices.push_back({i1.first.id, i1.second, vertex_source::created});
    vertices.push_back({i0.first.id, i0.second, vertex_source::created});
    for (Index i = i0.first.target.id + 1; i <= i1.first.target.id; ++i)
      vertices.push_back({Index(face[i]), i, vertex_source::original});

    auto v0 =
        loop::vertex<Index>{i0.first.id, i0.second, vertex_source::created};
    auto v1 =
        loop::vertex<Index>{i1.first.id, i1.second, vertex_source::created};
    if (v1 < v0)
      std::swap(v0, v1);
    if (v0 != v1) {
      _all_edges.push_back(v0);
      _all_edges.push_back(v1);
    }
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

  auto add_edge(vertex<Index> v0, vertex<Index> v1) {
    if (v1 < v0)
      std::swap(v0, v1);
    if (v0 == v1)
      return;
    _all_edges.push_back(v0);
    _all_edges.push_back(v1);
    auto edge = std::array<vertex<Index>, 2>{v0, v1};
    auto it = std::lower_bound(_base_loop_edges.begin(), _base_loop_edges.end(),
                               edge);
    if (!(it != _base_loop_edges.end() && *it == edge)) {
      _edges.push_back(v0);
      _edges.push_back(v1);
    }
  }

  template <typename Range, typename Policy, typename F, typename F1>
  auto extract_edges(const Range &intersections, const tf::points<Policy> &,
                     tf::intersect::simple_intersection<Index>,
                     const F &get_flat_id, const F1 &) {
    if (intersections.size() != 2)
      return;

    add_edge({intersections[0].id, get_flat_id(intersections[0]),
              vertex_source::created},
             {intersections[1].id, get_flat_id(intersections[1]),
              vertex_source::created});
  }

  template <typename Range, typename Policy, typename T, typename F,
            typename F1>
  auto extract_edges_impl(const Range &intersections,
                          const tf::points<Policy> &intersection_points, T,
                          const F &get_flat_id, const F1 &describe_other) {
    auto it = intersections.begin();
    auto end = intersections.end();
    while (it != end) {
      auto next = std::find_if(it + 1, end, [it](const auto &x) {
        return x.object_other != it->object_other;
      });
      if (next - it == 2) {
        add_edge(
            {it->id, get_flat_id(*it), vertex_source::created},
            {(it + 1)->id, get_flat_id(*(it + 1)), vertex_source::created});
      } else if (next - it > 2) {
        auto [is_coplanar, other_poly_size] = describe_other(it->object_other);
        if (is_coplanar) {
          extract_edges_coplanar(it, next, other_poly_size, intersection_points,
                                 get_flat_id);
        } else {
          extract_edges(it, next, intersection_points, get_flat_id);
        }
      }
      it = next;
    }
    make_edges_unique();
    make_all_edges_unique();
  }

  template <typename Range, typename Policy, typename F, typename F1>
  auto extract_edges(const Range &intersections,
                     const tf::points<Policy> &intersection_points,
                     tf::intersect::tagged_intersection<Index> ins,
                     const F &get_flat_id, const F1 &describe_other) {
    return extract_edges_impl(intersections, intersection_points, ins,
                              get_flat_id, describe_other);
  }

  template <typename Range, typename Policy, typename F, typename F1>
  auto extract_edges(const Range &intersections,
                     const tf::points<Policy> &intersection_points,
                     tf::intersect::intersection<Index> ins,
                     const F &get_flat_id, const F1 &describe_other) {
    return extract_edges_impl(intersections, intersection_points, ins,
                              get_flat_id, describe_other);
  }

  template <typename Iterator, typename Policy, typename F>
  auto extract_edges(Iterator begin, Iterator end,
                     const tf::points<Policy> &intersection_points,
                     const F &get_flat_id) {
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
      auto i = *it;
      auto flat_id = get_flat_id(*it);
      auto pt = intersection_points[i.id];
      _work_buffer.push_back({i.target, i.id, flat_id, RealT(0)});
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
    for (auto [a, b] : tf::make_slide_range<2>(_work_buffer)) {
      add_edge({a.id, a.intersection_id, vertex_source::created},
               {b.id, b.intersection_id, vertex_source::created});
    }
  }

  template <typename Iterator, typename Policy, typename F>
  auto extract_edges_coplanar(Iterator begin, Iterator end,
                              std::size_t other_poly_size,
                              const tf::points<Policy> &intersection_points,
                              const F &get_flat_id) {
    _work_buffer.clear();

    for (auto it = begin; it != end; ++it) {
      if (it->target_other.label == tf::topo_type::face)
        continue;

      auto flat_id = get_flat_id(*it);

      if (it->target_other.label == tf::topo_type::edge) {
        _work_buffer.push_back(
            {{Index(it->target_other.id), tf::topo_type::edge}, it->id, flat_id,
             RealT(0)});
      } else if (it->target_other.label == tf::topo_type::vertex) {
        auto v = it->target_other.id;
        auto prev_edge = (v - 1 + other_poly_size) % other_poly_size;
        _work_buffer.push_back(
            {{Index(prev_edge), tf::topo_type::edge}, it->id, flat_id,
             std::numeric_limits<RealT>::max()});
        _work_buffer.push_back({{Index(v), tf::topo_type::edge}, it->id,
                                flat_id, std::numeric_limits<RealT>::lowest()});
      }
    }

    if (_work_buffer.size() < 2)
      return;

    std::sort(_work_buffer.begin(), _work_buffer.end(),
              [](const auto &x, const auto &y) {
                return x.target.id < y.target.id;
              });

    auto edge_begin = _work_buffer.begin();
    while (edge_begin != _work_buffer.end()) {
      auto edge_id = edge_begin->target.id;
      auto edge_end =
          std::find_if(edge_begin, _work_buffer.end(),
                       [edge_id](const auto &x) { return x.target.id != edge_id; });

      auto count = std::distance(edge_begin, edge_end);

      if (count == 2) {
        add_edge({edge_begin->id, edge_begin->intersection_id,
                  vertex_source::created},
                 {(edge_begin + 1)->id, (edge_begin + 1)->intersection_id,
                  vertex_source::created});
      } else if (count > 2) {
        auto origin = intersection_points[edge_begin->id];
        auto dir = intersection_points[(edge_end - 1)->id] - origin;

        for (auto it = edge_begin; it != edge_end; ++it) {
          if (it->t != std::numeric_limits<RealT>::max() &&
              it->t != std::numeric_limits<RealT>::lowest()) {
            it->t = tf::dot(intersection_points[it->id] - origin, dir);
          }
        }

        std::sort(edge_begin, edge_end, [](const auto &x, const auto &y) {
          return std::make_pair(x.t, x.id) < std::make_pair(y.t, y.id);
        });

        for (auto it = edge_begin; it + 1 != edge_end; ++it) {
          add_edge({it->id, it->intersection_id, vertex_source::created},
                   {(it + 1)->id, (it + 1)->intersection_id,
                    vertex_source::created});
        }
      }

      edge_begin = edge_end;
    }
  }

  auto make_edges_unique() {
    auto es = tf::make_blocked_range<2>(_edges);
    std::sort(es.begin(), es.end());
    auto n = (std::unique(es.begin(), es.end()) - es.begin()) * 2;
    _edges.erase_till_end(_edges.begin() + n);
  }

  auto make_all_edges_unique() {
    auto es = tf::make_blocked_range<2>(_all_edges);
    std::sort(es.begin(), es.end());
    auto n = (std::unique(es.begin(), es.end()) - es.begin()) * 2;
    _all_edges.erase_till_end(_all_edges.begin() + n);
  }

  template <typename Range0, typename Range1, typename Policy, typename Range2,
            typename F>
  auto extract_base_loop_from_intersections(
      const Range0 &face, const Range1 &intersection_points,
      const tf::points<Policy> &mesh_points, const Range2 &intersections,
      const F &get_flat_id) {
    for (const auto &x : intersections)
      if (x.target.label != tf::topo_type::face)
        _work_buffer.push_back({x.target, x.id, get_flat_id(x), RealT(0)});
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
        _base_loop.push_back({Index(face[i]), i, vertex_source::original});
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
        _base_loop.push_back({Index(face[i]), i, vertex_source::original});
      while (it != it_end) {
        _base_loop.push_back(
            {it->id, it->intersection_id, vertex_source::created});
        ++it;
      }
      it = next_it;
    }
  }

  struct node_t {
    tf::intersect::intersection_target<Index> target;
    Index id;
    Index intersection_id;
    RealT t;
  };

  tf::buffer<node_t> _work_buffer;
  tf::buffer<vertex<Index>> _base_loop;
  tf::buffer<vertex<Index>> _edges;
  tf::buffer<vertex<Index>> _all_edges;
  tf::buffer<std::array<vertex<Index>, 2>> _base_loop_edges;
  tf::loop::cut_face_by_intersections<Index, RealT> _cf;
};

} // namespace tf::loop
