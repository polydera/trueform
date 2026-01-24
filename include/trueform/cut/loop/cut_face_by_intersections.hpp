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
#include "../../core/array_hash.hpp"
#include "../../core/buffer.hpp"
#include "../../core/contiguous_index_hash_map.hpp"
#include "../../core/index_hash_map.hpp"
#include "../../core/points_buffer.hpp"
#include "../../topology/hole_patcher.hpp"
#include "./face_split_by_edges.hpp"
#include "./vertex.hpp"

namespace tf::loop {
template <typename Index, typename RealT> class cut_face_by_intersections {
public:
  template <typename Range0, typename Policy, typename F>
  auto build(const Range0 &base_loop, const tf::edges<Policy> &edges,
             const F &get_point) {
    clear();
    tf::make_contiguous_index_hash_map(base_loop, _ihm, Index(0));
    tf::make_contiguous_index_hash_map(edges, _ihm,
                                       Index(_ihm.kept_ids().size()));

    _base_loop.reserve(base_loop.size());
    for (auto v : base_loop)
      _base_loop.push_back(_ihm.f()[v]);
    _edges.reserve(edges.size() * 2);
    for (auto edge : edges) {
      _edges.push_back(_ihm.f()[edge[0]]);
      _edges.push_back(_ihm.f()[edge[1]]);
    }
    _points.reserve(_ihm.kept_ids().size());
    for (auto v : _ihm.kept_ids())
      _points.push_back(get_point(v));
    /*std::cout << "map" << std::endl;*/
    /*for(auto [e, b]:_ihm.f()){*/
    /*  std::cout << "(" << int(e.source) << ", " << e.id << ", "*/
    /*              << e.intersection_index << "): " << b << std::endl;*/
    /*} */
    /**/
    /*std::cout << "base_loop" << std::endl;*/
    /*for(auto e:_base_loop)*/
    /*  std::cout << e << ", ";*/
    /*std::cout << std::endl;*/
    /*std::cout << "edges" << std::endl;*/
    /*for(auto [a, b]: tf::make_blocked_range<2>(_edges))*/
    /*  std::cout << a << ", " << b << std::endl;*/

    _fs.build(_base_loop, tf::make_edges(tf::make_blocked_range<2>(_edges)),
              tf::make_points(_points));
  }

  auto clear() {
    _ihm.clear();
    _fs.clear();
    _edges.clear();
    _base_loop.clear();
    _hp.clear();
    _points.clear();
  }

  auto extract(tf::buffer<Index> &offsets,
               tf::buffer<loop::vertex<Index>> &vertices) {
    if (_fs.faces().size() == 0) {
      // Fall back to base loop when cutting produces no valid faces
      write_face(_base_loop, vertices, offsets);
      return;
    }
    if (_fs.holes().size())
      extract_with_holes(vertices, offsets);
    else
      extract_without_holes(vertices, offsets);
  }

private:
  template <typename Range>
  auto write_face(const Range &face, tf::buffer<loop::vertex<Index>> &vertices,
                  tf::buffer<Index> &offsets) {
    offsets.push_back(vertices.size());
    for (auto id : face)
      vertices.push_back(_ihm.kept_ids()[id]);
  }

  auto extract_without_holes(tf::buffer<loop::vertex<Index>> &vertices,
                             tf::buffer<Index> &offsets) {
    for (const auto &face : _fs.faces()) {
      write_face(face, vertices, offsets);
    }
  }

  auto extract_with_holes(tf::buffer<loop::vertex<Index>> &vertices,
                          tf::buffer<Index> &offsets) {
    for (const auto &[face, hole_ids] :
         tf::zip(_fs.faces(), _fs.holes_for_faces())) {
      if (hole_ids.size()) {
        _hp.build(
            face,
            tf::make_faces(tf::make_indirect_range(hole_ids, _fs.holes())),
            tf::make_points(_points));
        write_face(_hp.face(), vertices, offsets);
      } else
        write_face(face, vertices, offsets);
    }
  }

  struct hash_t {
    auto operator()(const loop::vertex<Index> &v) const {
      return hash(std::array<Index, 2>{Index(v.source), Index(v.id)});
    }
    tf::array_hash<Index, 2> hash;
  };

  tf::index_hash_map<loop::vertex<Index>, Index, hash_t> _ihm;
  tf::loop::face_split_by_edges<Index, RealT> _fs;
  tf::hole_patcher<Index> _hp;
  tf::buffer<Index> _edges;
  tf::buffer<Index> _base_loop;
  tf::points_buffer<RealT, 2> _points;
};
} // namespace tf::loop
